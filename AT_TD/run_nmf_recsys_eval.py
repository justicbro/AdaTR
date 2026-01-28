
import os, math, inspect, argparse
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "8")
from types import SimpleNamespace
import numpy as np, pandas as pd
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

def load_movielens_100k(path: str):
    import pandas as pd
    df = pd.read_csv(path, sep="\t", names=["user","item","rating","ts"], engine="python")
    n_users = df["user"].max()
    n_items = df["item"].max()
    X = np.zeros((n_users, n_items), dtype=float)
    for _, row in df.iterrows():
        X[int(row["user"])-1, int(row["item"])-1] = float(row["rating"])
    return X

def make_synthetic(users=500, items=450, rank=20, obs_rate=0.12):
    W_true = rng.gamma(shape=1.0, scale=1.0, size=(users, rank))
    H_true = rng.gamma(shape=1.0, scale=1.0, size=(rank, items))
    X_full = W_true @ H_true
    X_scaled = 5 * (1 - np.exp(-X_full / (X_full.mean() + 1e-8)))
    ratings = np.clip(np.round(X_scaled), 1, 5)
    mask = rng.random(ratings.shape) < obs_rate
    X_sparse = np.where(mask, ratings, 0.0)
    return X_sparse

def build_matrix_from_indices(X_base, idx_keep):
    X_new = np.zeros_like(X_base)
    for u, i in idx_keep:
        X_new[u, i] = X_base[u, i]
    return X_new

def fit_nmf_and_predict(X_input, rank=10, max_iter=100, random_state=42):
    model = NMF(n_components=rank, init="nndsvda", max_iter=max_iter, random_state=random_state, solver="cd", beta_loss="frobenius")
    W = model.fit_transform(X_input)
    H = model.components_
    return W @ H

def rmse_at_pairs(X_hat, pairs, y_true):
    preds = np.array([X_hat[u, i] for u, i in pairs])
    return math.sqrt(mean_squared_error(y_true, preds))

def topk_metrics(X_hat, train_mat, test_pairs, K=10):
    from collections import defaultdict
    test_pos_by_user = defaultdict(set)
    for u, i in test_pairs:
        test_pos_by_user[int(u)].add(int(i))
    prec_list, rec_list = [], []
    for u, pos_items in test_pos_by_user.items():
        seen = set(np.argwhere(train_mat[u] > 0).reshape(-1).tolist())
        scores = X_hat[u].copy()
        scores[list(seen)] = -1e9
        topk = np.argpartition(scores, -K)[-K:]
        topk = topk[np.argsort(scores[topk])[::-1]]
        hits = len(set(topk.tolist()) & pos_items)
        prec_list.append(hits / K)
        rec_list.append(hits / max(1, len(pos_items)))
    return float(np.mean(prec_list)), float(np.mean(rec_list))

def add_gaussian_noise_on_observed(
    X_mat,
    idx_obs,
    sigma=0.35,
    epsilon=1.0,
    norm_type="l2",
    clip_min=0.0,
    clip_max=5.0,
):
    diff = np.zeros_like(X_mat, dtype=float)
    noise = rng.normal(loc=0.0, scale=sigma, size=len(idx_obs))
    for k, (u, i) in enumerate(idx_obs):
        diff[u, i] = noise[k]
    if epsilon > 0:
        if norm_type == "l2":
            norm = np.linalg.norm(diff)
            if norm > epsilon:
                diff *= epsilon / (norm + 1e-12)
        else:  # linf
            diff = np.clip(diff, -epsilon, epsilon)
    X_noisy = np.clip(X_mat + diff, clip_min, clip_max)
    return X_noisy

# --- replace load_user_attacks + try_attack with the following ---

def load_user_attacks(module_path):
    """Dynamically import user's attack module."""
    try:
        import importlib.util, importlib.machinery, sys, os
        if not os.path.exists(module_path):
            print(f"[WARN] attacks_py not found: {module_path}")
            return None, None
        loader = importlib.machinery.SourceFileLoader("user_attacks", module_path)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        user_attacks = importlib.util.module_from_spec(spec)
        loader.exec_module(user_attacks)
        return getattr(user_attacks, "optimize_perturbation", None), getattr(user_attacks, "optimize_perturbation_comparison", None)
    except Exception as e:
        print(f"[WARN] Could not import attacks due to: {e}")
        return None, None

def build_attack_config(args, X_train):
    # 设备选择：优先GPU
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    cfg = SimpleNamespace(
        epsilon   = getattr(args, "epsilon", 0.5),
        steps     = getattr(args, "steps", 20),
        step_size = getattr(args, "step_size", 0.05),
        norm      = getattr(args, "norm", "linf"),
        clip_min  = 0.0,
        clip_max  = 5.0,
        seed      = 42,
        shape     = X_train.shape,
        device    = device,

        inner_num_steps = getattr(args, "inner_num_steps", 200), # 内循环步数（若有）
        outer_num_steps = getattr(args, "outer_num_steps", 100), # 外循环步数（若有）
        inner_dis_num   = getattr(args, "inner_dis_num", 300),   # 内循环discriminator步数（若有）
        outer_dis_num   = getattr(args, "outer_dis_num", 10),   # 外循环discriminator步数（若有）    

        inner_tol = getattr(args, "inner_tol", 1e-9), # 内循环收敛阈值（若有）
    )
    # 常见别名，兼容不同实现
    cfg.eps    = cfg.epsilon
    cfg.lr     = cfg.step_size
    cfg.n_iter = cfg.steps
    cfg.p      = np.inf if str(cfg.norm).lower() == "linf" else 2
    return cfg

def _to_torch_if_needed(X):
    """If attack expects torch.Tensor, convert; else return numpy."""
    import inspect
    import numpy as np
    try:
        import torch
    except Exception:
        torch = None

    # Heuristic: if torch is available, we pass torch.Tensor; most AT实现在torch里
    if torch is not None:
        return torch.from_numpy(X.astype("float32"))
    return X

def _to_numpy(X_adv):
    """Convert possible torch result back to numpy."""
    try:
        import torch
        if isinstance(X_adv, torch.Tensor):
            return X_adv.detach().cpu().numpy()
    except Exception:
        pass
    return X_adv

def try_attack(fn, X_in, rank, config):
    """优先用 (X, rank, config) 调用；确保 X 与内部新建张量在同一设备。"""
    if fn is None:
        return X_in.copy(), False

    import inspect, numpy as np
    X_np = X_in.astype(np.float32)

    # 把输入搬到 config.device
    try:
        import torch
        device = torch.device(getattr(config, "device", "cpu"))
        X_arg = torch.from_numpy(X_np).to(device)

        # PyTorch>=2.0 可设置默认设备，避免函数里新建张量落在CPU
        if hasattr(torch, "set_default_device"):
            torch.set_default_device(device)
    except Exception:
        # 没有torch就按numpy跑（你这条路径基本不会走）
        X_arg = X_np
        device = None

    try:
        sig = inspect.signature(fn)
        n = len(sig.parameters)
        if n >= 3:
            X_adv = fn(X_arg, rank, config)
        elif n == 2:
            X_adv = fn(X_arg, rank)
        else:
            X_adv = fn(X_arg)

        # 兼容返回 (X_adv, extra) 的写法
        if isinstance(X_adv, (tuple, list)):
            X_adv = X_adv[0]

        # 转回 numpy
        try:
            import torch
            if isinstance(X_adv, torch.Tensor):
                X_adv = X_adv.detach().cpu().numpy()
        except Exception:
            pass

        if X_adv.shape != X_in.shape:
            raise ValueError(f"Attack returned shape {X_adv.shape}, expected {X_in.shape}")

        X_adv = np.clip(X_adv, 0.0, 5.0)
        return X_adv, True
    except Exception as e:
        print(f"[WARN] attack failed: {e}")
        return X_in.copy(), False

# 评测: 每用户 1 正例 + 99 负例
def hr_ndcg_at_k(X_hat, train_mat, test_pos, K=10, n_neg=99):
    import math
    U, I = X_hat.shape
    all_items = np.arange(I)
    hr_list, ndcg_list = [], []
    rng = np.random.default_rng(123)
    for u, pos_item in test_pos.items():
        seen = set(np.argwhere(train_mat[u] > 0).reshape(-1).tolist())
        # 采 99 个未见负例
        cand_neg = np.setdiff1d(all_items, np.array(list(seen | {pos_item})), assume_unique=False)
        if len(cand_neg) < n_neg: 
            continue
        neg_sample = rng.choice(cand_neg, size=n_neg, replace=False)
        candidates = np.concatenate([[pos_item], neg_sample])

        scores = X_hat[u, candidates]
        ranking = np.argsort(-scores)  # 降序
        topk = candidates[ranking[:K]]

        # HR@K
        hit = int(pos_item in topk)
        hr_list.append(hit)

        # NDCG@K
        # 只有一个正例，DCG = 1/log2(1+rank_of_pos)
        try:
            rank_pos = np.where(topk == pos_item)[0]
            if len(rank_pos) > 0:
                dcg = 1.0 / math.log2(2 + rank_pos[0])
            else:
                dcg = 0.0
        except:
            dcg = 0.0
        ndcg_list.append(dcg)  # IDCG=1

    return float(np.mean(hr_list)), float(np.mean(ndcg_list))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--movielens_path", type=str, default="", help="Path to MovieLens 100K u.data (tab-separated)")
    ap.add_argument("--attacks_py", type=str, default="attacks.py", help="Path to your attacks .py containing optimize_perturbation* fns")
    ap.add_argument("--rank", type=int, default=20)
    ap.add_argument("--max_iter", type=int, default=100)
    ap.add_argument("--sigma", type=float, default=0.35)
    # ap.add_argument("--epsilon", type=float, default=100) # for movielens
    ap.add_argument("--epsilon", type=float, default=10) # for synthetic
    ap.add_argument("--obs_rate", type=float, default=0.12, help="Observation rate for synthetic data (fraction observed)")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--step_size", type=float, default=0.05)
    ap.add_argument("--norm", type=str, choices=["linf", "l2"], default="l2")
    ap.add_argument(
        "--run-linf",
        action="store_true",
        help="Also evaluate all attacks under L-infinity norm constraint.",
    )

    args = ap.parse_args()

    if args.movielens_path and os.path.exists(args.movielens_path):
        X = load_movielens_100k(args.movielens_path)
        dataset_name = "MovieLens-100K"
    else:
        X = make_synthetic(obs_rate=args.obs_rate)
        dataset_name = "Synthetic"

    obs_idx = np.argwhere(X > 0)
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(obs_idx, test_size=0.2, random_state=RANDOM_SEED)
    X_train = build_matrix_from_indices(X, train_idx)
    X_test_pairs = test_idx.copy()
    y_test = np.array([X[u, i] for u, i in X_test_pairs])

    POS = 4
    keep = []
    for (u, i) in X_test_pairs:
        if X[u, i] >= POS:
            keep.append((u, i))
    X_test_pairs = np.array(keep, dtype=int)
    y_test = np.array([X[u, i] for (u, i) in X_test_pairs])

    opt_perturb, opt_perturb_cmp = load_user_attacks(args.attacks_py)

    

    results = []

    # Clean
    X_hat = fit_nmf_and_predict(X_train, rank=args.rank, max_iter=args.max_iter)
    rmse = rmse_at_pairs(X_hat, X_test_pairs, y_test)
    p10, r10 = topk_metrics(X_hat, X_train, X_test_pairs)
    results.append({"condition":"Clean (no noise)", "RMSE": rmse, "Precision@10": p10, "Recall@10": r10})

    def eval_and_append(tag, X_matrix):
        X_hat = fit_nmf_and_predict(X_matrix, rank=args.rank, max_iter=args.max_iter)
        rmse = rmse_at_pairs(X_hat, X_test_pairs, y_test)
        p10, r10 = topk_metrics(X_hat, X_train, X_test_pairs)
        results.append({"condition": tag, "RMSE": rmse, "Precision@10": p10, "Recall@10": r10})

    # Norm variants
    norm_variants = [(args.norm, "F" if args.norm == "l2" else "L∞")]
    if args.run_linf and args.norm != "linf":
        norm_variants.append(("linf", "L∞"))
    seen = set()
    unique_norms = []
    for nm, label in norm_variants:
        if nm not in seen:
            unique_norms.append((nm, label))
            seen.add(nm)

    total_entries = max(1, np.count_nonzero(X_train))
    epsilon_f = float(args.epsilon)
    epsilon_linf = epsilon_f / math.sqrt(total_entries)

    for norm_type, label in unique_norms:
        eps = epsilon_f if norm_type == "l2" else epsilon_linf
        sigma = args.sigma
        tag_suffix = f" ({label})"

        # Gaussian
        Xg = add_gaussian_noise_on_observed(
            X_train,
            train_idx,
            sigma=sigma,
            epsilon=eps,
            norm_type=norm_type,
        )
        eval_and_append(f"Gaussian noise{tag_suffix}", Xg)

        # Build config for attacks
        config = build_attack_config(args, X_train)
        config.norm = norm_type
        config.epsilon = eps
        config.eps = eps
        config.p = np.inf if norm_type == "linf" else 2

        # ATNMF
        Xat, ok = try_attack(opt_perturb_cmp, X_train, args.rank, config)
        label_at = f"ATNMF (optimize_perturbation_comparison){tag_suffix}"
        if not ok:
            label_at += " [not executed]"
        eval_and_append(label_at, Xat)

        # AdaNMF
        Xad, ok = try_attack(opt_perturb, X_train, args.rank, config)
        label_ad = f"AdaNMF (optimize_perturbation){tag_suffix}"
        if not ok:
            label_ad += " [not executed]"
        eval_and_append(label_ad, Xad)

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    out_path = "nmf_recsys_results2.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
