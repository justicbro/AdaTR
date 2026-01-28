#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MovieLens-100K Top-K 推荐 + 扰动攻击评测
- 数据：MovieLens 100K（自动下载/解压）
- 训练：简单 MF (SGD)
- 指标：HR@K, NDCG@K (leave-one-out)
- 攻击：调用用户提供的 optimize_perturbation(X, tr_cores, config) 返回 E，做数据中毒
运行：python recsys_attack_movielens.py --help
"""
import os
import io
import zipfile
import urllib.request
import argparse
from typing import Tuple, Dict, Optional, List, Set
from AT_TR_plus import optimize_perturbation

import numpy as np

# ====================== Utils ======================

def set_seed(seed=42):
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    return rng

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# ====================== Data ======================

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

def download_ml100k(data_dir: str) -> str:
    ensure_dir(data_dir)
    zip_path = os.path.join(data_dir, "ml-100k.zip")
    target_dir = os.path.join(data_dir, "ml-100k")
    if os.path.isdir(target_dir) and os.path.isfile(os.path.join(target_dir, "u.data")):
        return target_dir
    if not os.path.isfile(zip_path):
        print(f"[INFO] downloading {ML100K_URL} -> {zip_path}")
        urllib.request.urlretrieve(ML100K_URL, zip_path)
    print("[INFO] extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    return target_dir

def load_ml100k_interactions(data_dir: str, implicit_threshold: Optional[float] = 4.0) -> np.ndarray:
    """
    返回稀疏评分矩阵 X ∈ R^{U×I}，未观测=0；若 implicit_threshold 不为 None，则评分>=threshold 置为1，其他为0（隐式反馈）。
    """
    path = os.path.join(data_dir, "u.data")
    # u.data: user item rating timestamp (tab-separated), user∈[1,943], item∈[1,1682]
    users, items, ratings = [], [], []
    with open(path, "r") as f:
        for line in f:
            u, i, r, _ = line.strip().split('\t')
            users.append(int(u) - 1)
            items.append(int(i) - 1)
            ratings.append(float(r))
    U = max(users) + 1
    I = max(items) + 1
    X = np.zeros((U, I), dtype=np.float32)
    for u, i, r in zip(users, items, ratings):
        if implicit_threshold is None:
            X[u, i] = r
        else:
            X[u, i] = 1.0 if r >= implicit_threshold else 0.0
    return X

# ============ Split & Metrics ============

def leave_one_out_split(interactions: np.ndarray, rng, min_pos=1):
    U, I = interactions.shape
    train = interactions.copy()
    test_ui = []
    user_pos_train: List[Set[int]] = []
    for u in range(U):
        pos_items = np.where(interactions[u] > 0)[0]
        if len(pos_items) >= max(min_pos, 1):
            i_test = int(rng.choice(pos_items))
            test_ui.append((u, i_test))
            train[u, i_test] = 0.0  # remove one pos to test
            user_pos_train.append(set(np.where(train[u] > 0)[0]))
        else:
            test_ui.append((u, -1))
            user_pos_train.append(set())
    return train, np.array(test_ui, dtype=int), user_pos_train

def hr_ndcg_at_k(scores_row: np.ndarray, gt_item: int, excluded: set, K=10):
    if gt_item < 0:
        return 0.0, 0.0
    I = len(scores_row)
    mask = np.ones(I, dtype=bool)
    if excluded:
        mask[list(excluded)] = False
    idx = np.argsort(-scores_row[mask])[:K]
    cand_items = np.arange(I)[mask][idx]
    if gt_item in cand_items:
        rank = int(np.where(cand_items == gt_item)[0][0]) + 1  # 1-based
        hr = 1.0
        ndcg = 1.0 / np.log2(rank + 1)
    else:
        hr, ndcg = 0.0, 0.0
    return hr, ndcg

def evaluate_topk(pred_scores: np.ndarray, test_ui: np.ndarray, user_pos_train, K=10) -> Dict[str, float]:
    U = pred_scores.shape[0]
    hrs, ndcgs = [], []
    for u in range(U):
        _, gt = test_ui[u]
        hr, ndcg = hr_ndcg_at_k(pred_scores[u], gt, user_pos_train[u], K=K)
        hrs.append(hr); ndcgs.append(ndcg)
    return {"HR@{}".format(K): float(np.mean(hrs)), "NDCG@{}".format(K): float(np.mean(ndcgs))}

# ====================== MF Model ======================

class MF_SGD:
    def __init__(self, n_users, n_items, rank=64, lr=5e-3, reg=1e-2, n_epochs=10, rng=None):
        self.U = n_users; self.I = n_items
        self.rank = rank; self.lr = lr; self.reg = reg; self.n_epochs = n_epochs
        self.rng = rng or set_seed(42)
        self.P = self.rng.normal(0, 0.1, size=(self.U, rank))
        self.Q = self.rng.normal(0, 0.1, size=(self.I, rank))

    def fit(self, X: np.ndarray):
        us, is_ = np.where(X > 0)
        vals = X[us, is_]
        idx = np.arange(len(vals))
        if len(idx) == 0:
            return
        for _ in range(self.n_epochs):
            self.rng.shuffle(idx)
            for t in idx:
                u, i, r = us[t], is_[t], vals[t]
                pred = float(self.P[u] @ self.Q[i])
                err = pred - r
                gu = err * self.Q[i] + self.reg * self.P[u]
                gi = err * self.P[u] + self.reg * self.Q[i]
                self.P[u] -= self.lr * gu
                self.Q[i] -= self.lr * gi

    def predict_all(self) -> np.ndarray:
        return self.P @ self.Q.T

# ====================== Attack Hook ======================

def call_optimize_perturbation(X: np.ndarray, tr_cores=None, config: Optional[dict]=None) -> np.ndarray:
    """
    期望用户在同目录下提供 optimize_perturbation。若不可用，回退到微量高斯噪声。
    """
    try:
        # from your_pkg import optimize_perturbation  # 若已安装为包
        E_opt, TN_G, loss_ours = optimize_perturbation(X, tr_cores, config)  # noqa: F821
        if E_opt.shape != X.shape:
            raise ValueError("E shape mismatch")
        return E_opt
    except Exception as e:
        print("[WARN] optimize_perturbation 不可用或失败：", repr(e))
        rng = set_seed(123)
        return rng.normal(0, 0.02 * max(1.0, X.std() + 1e-6), size=X.shape)

# ====================== Pipeline ======================

def run_experiment(data_root: str,
                   implicit_threshold: float = 4.0,
                   rank: int = 64, K: int = 10,
                   n_epochs: int = 15, lr: float = 5e-3, reg: float = 1e-2,
                   clip_min: float = 0.0, clip_max: float = 1.0,
                   tr_cores=None, config: Optional[dict]=None,
                   seed: int = 42) -> Dict[str, Dict[str, float]]:
    rng = set_seed(seed)

    # 1) Data
    ml_dir = download_ml100k(data_root)
    X = load_ml100k_interactions(ml_dir, implicit_threshold=implicit_threshold)
    print(f"[INFO] X shape={X.shape}, positives={int((X>0).sum())}")

    # 2) Leave-one-out
    X_train, test_ui, user_pos_train = leave_one_out_split(X, rng)
    print(f"[INFO] leave-one-out done. test users={len(test_ui)}")

    # 3) Train baseline MF
    mf = MF_SGD(X.shape[0], X.shape[1], rank=rank, lr=lr, reg=reg, n_epochs=n_epochs, rng=rng)
    mf.fit(X_train)
    scores_clean = mf.predict_all()
    metrics_clean = evaluate_topk(scores_clean, test_ui, user_pos_train, K=K)
    print("[CLEAN]", metrics_clean)

    # 4) Attack: poison training matrix
    E = call_optimize_perturbation(X_train, tr_cores=tr_cores, config=config)
    X_poison = np.clip(X_train + E, clip_min, clip_max)

    # 可选：稀疏化（只在原有观测位置扰动）——若希望严格遵守观测掩码，请启用：
    # mask_obs = (X_train > 0)
    # X_poison = X_train.copy()
    # X_poison[mask_obs] = np.clip(X_train[mask_obs] + E[mask_obs], clip_min, clip_max)

    # 5) Retrain on poisoned
    mf_p = MF_SGD(X.shape[0], X.shape[1], rank=rank, lr=lr, reg=reg, n_epochs=n_epochs, rng=rng)
    mf_p.fit(X_poison)
    scores_poison = mf_p.predict_all()
    metrics_poison = evaluate_topk(scores_poison, test_ui, user_pos_train, K=K)
    print("[POISON]", metrics_poison)

    return {"clean": metrics_clean, "poison": metrics_poison}

# ====================== CLI ======================

def main():
    parser = argparse.ArgumentParser(description="MovieLens-100K 推荐攻击评测")
    parser.add_argument("--data_root", type=str, default="./data", help="数据根目录（将自动下载到此处）")
    parser.add_argument("--implicit_threshold", type=float, default=4.0, help="评分>=阈值 视为正反馈；设为-1可用显式评分")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--reg", type=float, default=1e-2)
    parser.add_argument("--clip_min", type=float, default=0.0)
    parser.add_argument("--clip_max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    implicit_threshold = None if args.implicit_threshold < 0 else args.implicit_threshold
    results = run_experiment(
        data_root=args.data_root,
        implicit_threshold=implicit_threshold,
        rank=args.rank, K=args.K,
        n_epochs=args.epochs, lr=args.lr, reg=args.reg,
        clip_min=args.clip_min, clip_max=args.clip_max,
        tr_cores=None, config=None, seed=args.seed
    )
    print("=== Results ===")
    print(results)

if __name__ == "__main__":
    main()
