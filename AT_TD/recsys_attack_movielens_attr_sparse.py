#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML-100K Top-K 推荐 + ATR 攻击（设备透传 + 稀疏扰动比例 edit_ratio）
- 只在训练集中已有正例处扰动；可选仅修改其中的 p%（edit_ratio）
- L_inf 预算 eps + 方向约束（默认只减弱正例）
- 设备 device 可选 auto/cuda/cpu，并透传给你的 TR 实现
"""
import os, zipfile, urllib.request, argparse, importlib, importlib.util
import numpy as np
from typing import Optional, Dict, List, Set, Tuple

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

def set_seed(seed=42):
    rng = np.random.RandomState(seed); np.random.seed(seed); return rng

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def download_ml100k(data_dir: str) -> str:
    ensure_dir(data_dir)
    z = os.path.join(data_dir, "ml-100k.zip")
    tgt = os.path.join(data_dir, "ml-100k")
    if os.path.isdir(tgt) and os.path.isfile(os.path.join(tgt,"u.data")):
        return tgt
    if not os.path.isfile(z):
        print(f"[INFO] downloading {ML100K_URL} -> {z}")
        urllib.request.urlretrieve(ML100K_URL, z)
    print("[INFO] extracting...")
    with zipfile.ZipFile(z, "r") as zf: zf.extractall(data_dir)
    return tgt

def load_ml100k_interactions(data_dir: str, implicit_threshold: Optional[float]=4.0) -> np.ndarray:
    path = os.path.join(data_dir, "u.data")
    users, items, ratings = [], [], []
    with open(path, "r") as f:
        for line in f:
            u,i,r,_ = line.strip().split('\t')
            users.append(int(u)-1); items.append(int(i)-1); ratings.append(float(r))
    U = max(users)+1; I = max(items)+1
    X = np.zeros((U,I), dtype=np.float32)
    if implicit_threshold is None:
        for u,i,r in zip(users,items,ratings): X[u,i] = r
    else:
        thr = float(implicit_threshold)
        for u,i,r in zip(users,items,ratings): X[u,i] = 1.0 if r>=thr else 0.0
    return X

def leave_one_out_split(X: np.ndarray, rng, min_pos=1):
    U,I = X.shape
    train = X.copy(); test_ui=[]; user_pos_train: List[Set[int]]=[]
    for u in range(U):
        pos = np.where(X[u]>0)[0]
        if len(pos) >= max(1,min_pos):
            it = int(rng.choice(pos)); test_ui.append((u,it)); train[u,it]=0.0
            user_pos_train.append(set(np.where(train[u]>0)[0]))
        else:
            test_ui.append((u,-1)); user_pos_train.append(set())
    return train, np.array(test_ui, dtype=int), user_pos_train

def hr_ndcg_at_k(scores_row: np.ndarray, gt_item: int, excluded: set, K=10):
    if gt_item < 0: return 0.0, 0.0
    I = len(scores_row); mask = np.ones(I, dtype=bool)
    if excluded: mask[list(excluded)] = False
    idx = np.argsort(-scores_row[mask])[:K]
    cand = np.arange(I)[mask][idx]
    if gt_item in cand:
        rank = int(np.where(cand==gt_item)[0][0]) + 1
        return 1.0, 1.0/np.log2(rank+1)
    return 0.0, 0.0

def evaluate_topk(S: np.ndarray, test_ui: np.ndarray, user_pos_train, K=10)->Dict[str,float]:
    hrs, ndcgs = [], []
    for u in range(S.shape[0]):
        _, gt = test_ui[u]
        hr, ndcg = hr_ndcg_at_k(S[u], gt, user_pos_train[u], K=K)
        hrs.append(hr); ndcgs.append(ndcg)
    return {"HR@{}".format(K): float(np.mean(hrs)), "NDCG@{}".format(K): float(np.mean(ndcgs))}

class MF_SGD:
    def __init__(self, U,I, rank=64, lr=5e-3, reg=1e-2, n_epochs=15, rng=None):
        self.U, self.I, self.rank = U, I, rank
        self.lr, self.reg, self.n_epochs = lr, reg, n_epochs
        self.rng = rng or set_seed(42)
        self.P = self.rng.normal(0,0.1,size=(U,rank)); self.Q = self.rng.normal(0,0.1,size=(I,rank))
    def fit(self, X: np.ndarray):
        us, is_ = np.where(X>0); vals = X[us,is_]; idx = np.arange(len(vals))
        if len(idx)==0: return
        for _ in range(self.n_epochs):
            self.rng.shuffle(idx)
            for t in idx:
                u,i,r = us[t], is_[t], vals[t]
                pred = float(self.P[u] @ self.Q[i]); err = pred - r
                gu = err*self.Q[i] + self.reg*self.P[u]; gi = err*self.P[u] + self.reg*self.Q[i]
                self.P[u] -= self.lr*gu; self.Q[i] -= self.lr*gi
    def predict_all(self): return self.P @ self.Q.T

# ---------- Attack Adapter (device + ranks 透传) ----------
def call_attr_optimize(X_train_np: np.ndarray,
                       attack_module: str,
                       eps: float, atr_lr: float, outer_steps: int, inner_steps: int, inner_tol: float,
                       ranks: Tuple[int,int,int]=(5,5,5),
                       device_choice: str="auto") -> np.ndarray:
    """
    适配你的 ATR 接口：
    - numpy(U,I) -> torch(U,I,1)；放到 device
    - tr_cores = initialize_tr_cores(shape, ranks+rank0)
    - config = OptimizationConfig(...)
    - 返回 numpy(U,I) 的 E
    """
    torch = importlib.import_module("torch")
    tl = importlib.import_module("tensorly"); tl.set_backend("pytorch")
    # 解析 device
    if device_choice == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device_choice in ("cuda","cpu"):
        device = device_choice
    else:
        device = "cpu"
    # TR 模块
    tr_mod_name = f"{attack_module}.tr"
    if importlib.util.find_spec(tr_mod_name):
        tr = importlib.import_module(tr_mod_name)
    else:
        tr = importlib.import_module("TR_func")
    mod = importlib.import_module(attack_module)
    opt_fun = getattr(mod, "optimize_perturbation")
    OptCfg = getattr(mod, "OptimizationConfig")

    # 数据与 cores
    X_t = torch.tensor(X_train_np, dtype=torch.float32, device=device).unsqueeze(-1)  # (U,I,1)
    ranks_list = list(ranks) + [ranks[0]]
    tr_cores = tr.initialize_tr_cores(list(X_t.shape), ranks_list, device=device)
    cfg = OptCfg(epsilon=float(eps), lr=float(atr_lr),
                 outer_num_steps=int(outer_steps), inner_num_steps=int(inner_steps),
                 outer_dis_num=max(1, outer_steps//10 or 1), inner_dis_num=10, inner_tol=float(inner_tol))
    # 调你的优化器
    E_t, TN_G, _ = opt_fun(X_t, tr_cores, cfg)
    if hasattr(E_t, "detach"): E_t = E_t.detach()
    E_np = E_t.squeeze(-1).to("cpu").numpy().astype(np.float32)
    if E_np.shape != X_train_np.shape:
        raise RuntimeError(f"E shape mismatch: got {E_np.shape}, expected {X_train_np.shape}")
    return E_np

def project_linf(E: np.ndarray, eps: float) -> np.ndarray:
    if eps is None or eps <= 0: return E
    return np.clip(E, -eps, eps)

# ---------- Pipeline with sparse edit_ratio ----------
def run_experiment(data_root: str,
                   implicit_threshold: float = 4.0,
                   rank: int = 64, K: int = 10,
                   n_epochs: int = 15, lr: float = 5e-3, reg: float = 1e-2,
                   clip_min: float = 0.0, clip_max: float = 1.0,
                   alpha: float = 1.0, eps: float = 0.2,
                   attack_module: Optional[str]=None,
                   atr_lr: float = 0.01, atr_outer: int = 5, atr_inner: int = 1000, atr_tol: float = 1e-8,
                   ranks: Tuple[int,int,int]=(5,5,5),
                   device: str = "auto",
                   edit_ratio: float = 1.0,     # 只改动 p% 的观测正例；默认 1.0 全改
                   seed: int=42):
    rng = set_seed(seed)
    # Data
    ml_dir = download_ml100k(data_root)
    X = load_ml100k_interactions(ml_dir, implicit_threshold=implicit_threshold)
    print(f"[INFO] X shape={X.shape}, positives={(X>0).sum()}")
    # Split
    X_train, test_ui, user_pos_train = leave_one_out_split(X, rng)
    # Baseline
    mf = MF_SGD(*X.shape, rank=rank, lr=lr, reg=reg, n_epochs=n_epochs, rng=rng)
    mf.fit(X_train); S_clean = mf.predict_all()
    m_clean = evaluate_topk(S_clean, test_ui, user_pos_train, K=K)
    print("[CLEAN]", m_clean)
    # Attack E
    if attack_module:
        try:
            E = call_attr_optimize(X_train, attack_module,
                                   eps=eps, atr_lr=atr_lr, outer_steps=atr_outer,
                                   inner_steps=atr_inner, inner_tol=atr_tol,
                                   ranks=ranks, device_choice=device)
            print("[WARN] 调用你的 optimize_perturbation 成功：", repr(e))
        except Exception as e:
            print("[WARN] 调用你的 optimize_perturbation 失败：", repr(e))
            rng2 = set_seed(123); E = rng2.normal(0, 0.01*max(1.0, X_train.std()+1e-6), size=X_train.shape)
    else:
        rng2 = set_seed(123); E = rng2.normal(0, 0.01*max(1.0, X_train.std()+1e-6), size=X_train.shape)
    # Mask of observed positives
    obs_mask = (X_train > 0)
    # 稀疏子集：只改动其中的 p%
    if edit_ratio < 1.0 and edit_ratio > 0.0:
        pos_idx = np.argwhere(obs_mask)  # N x 2
        N = pos_idx.shape[0]
        k = max(1, int(np.floor(edit_ratio * N)))
        sel = rng.choice(N, size=k, replace=False)
        chosen = pos_idx[sel]
        edit_mask = np.zeros_like(obs_mask, dtype=bool)
        edit_mask[chosen[:,0], chosen[:,1]] = True
    else:
        edit_mask = obs_mask.copy()
    # 方向约束 + 预算（仅减弱）
    E_masked = np.zeros_like(E, dtype=np.float32)
    E_masked[edit_mask] = -np.abs(E[edit_mask])
    E_masked = project_linf(E_masked, eps=eps)
    # 应用扰动
    X_poison = X_train.copy()
    X_poison[edit_mask] = np.clip(X_train[edit_mask] + alpha * E_masked[edit_mask], clip_min, clip_max)
    # 未选择的位置保持原值；未观测保持0
    X_poison[~obs_mask] = 0.0
    # Retrain
    mf2 = MF_SGD(*X.shape, rank=rank, lr=lr, reg=reg, n_epochs=n_epochs, rng=rng)
    mf2.fit(X_poison); S_poison = mf2.predict_all()
    m_poison = evaluate_topk(S_poison, test_ui, user_pos_train, K=K)
    print("[POISON]", m_poison)
    # 附加报告
    changed = int(edit_mask.sum())
    print(f"[INFO] edited entries: {changed} / {(obs_mask).sum()}  ({changed/ max(1,(obs_mask).sum()):.2%})")
    return {"clean": m_clean, "poison": m_poison}

def main():
    ap = argparse.ArgumentParser(description="ML-100K 推荐攻击（ATR 集成 + 稀疏 edit_ratio + 设备控制）")
    ap.add_argument("--data_root", type=str, default="./data")
    ap.add_argument("--implicit_threshold", type=float, default=4.0)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--reg", type=float, default=1e-2)
    ap.add_argument("--clip_min", type=float, default=0.0)
    ap.add_argument("--clip_max", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=1.0, help="扰动缩放系数")
    ap.add_argument("--eps", type=float, default=1, help="L_inf 预算（对被选中的观测条目）")
    ap.add_argument("--attack_module", type=str, default=None, help="包含 optimize_perturbation 与 OptimizationConfig 的模块名（如 AT_TD）")
    ap.add_argument("--atr_lr", type=float, default=0.01)
    ap.add_argument("--atr_outer", type=int, default=5)
    ap.add_argument("--atr_inner", type=int, default=1)
    ap.add_argument("--atr_tol", type=float, default=1e-8)
    ap.add_argument("--r1", type=int, default=5); ap.add_argument("--r2", type=int, default=5); ap.add_argument("--r3", type=int, default=5)
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cuda","cpu"])
    ap.add_argument("--edit_ratio", type=float, default=1.0, help="仅修改观测正例中的比例 p∈(0,1]")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    thr = None if args.implicit_threshold < 0 else args.implicit_threshold
    res = run_experiment(
        data_root=args.data_root,
        implicit_threshold=thr,
        rank=args.rank, K=args.K,
        n_epochs=args.epochs, lr=args.lr, reg=args.reg,
        clip_min=args.clip_min, clip_max=args.clip_max,
        alpha=args.alpha, eps=args.eps,
        attack_module=args.attack_module,
        atr_lr=args.atr_lr, atr_outer=args.atr_outer, atr_inner=args.atr_inner, atr_tol=args.atr_tol,
        ranks=(args.r1, args.r2, args.r3),
        device=args.device,
        edit_ratio=args.edit_ratio,
        seed=args.seed
    )
    print("=== Results ==="); print(res)

if __name__ == "__main__":
    main()
