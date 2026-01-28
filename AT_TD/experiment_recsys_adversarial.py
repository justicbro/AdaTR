
"""
End-to-end recommendation robustness experiment comparing 5 variants:
1) Clean (no noise)
2) Gaussian noise
3) ATTR (optimize_perturbation_comparison)
4) FAG-AdaTR (optimize_perturbation_simplifed)
5) AdaTR (optimize_perturbation)

Usage:
  python experiment_recsys_adversarial.py --data synthetic
  # or point to MovieLens 100K ratings file:
  python experiment_recsys_adversarial.py --data ml100k --ml100k_path /path/to/u.data
"""

import argparse
import numpy as np
import pandas as pd
import TR_func as tr
# 在文件头部或函数顶部补一行：
from types import SimpleNamespace

from typing import Tuple, Dict, Callable, Optional

# ---------- (copied minimal MF + metrics from the notebook) ----------
def train_mf_sgd(train_ui, mask_train, n_factors=40, lr=0.01, reg=0.05, n_epochs=25, seed=42):
    rng = np.random.default_rng(seed)
    n_users, n_items = train_ui.shape
    U = 0.1 * rng.standard_normal((n_users, n_factors))
    V = 0.1 * rng.standard_normal((n_items, n_factors))
    user_idx, item_idx = np.where(mask_train)
    nnz = len(user_idx)
    for _ in range(n_epochs):
        order = rng.permutation(nnz)
        for t in order:
            u = user_idx[t]; i = item_idx[t]; r = train_ui[u, i]
            pred = U[u] @ V[i]
            e = pred - r
            U[u] -= lr * (e * V[i] + reg * U[u])
            V[i] -= lr * (e * U[u] + reg * V[i])
    return U, V

def predict_all(U, V): return U @ V.T

def rmse(pred, truth, mask):
    diff = (pred - truth)[mask]
    return float(np.sqrt(np.mean(diff * diff))) if np.any(mask) else float("nan")

def mae(pred, truth, mask):
    diff = np.abs((pred - truth)[mask])
    return float(np.mean(diff)) if np.any(mask) else float("nan")

def make_synthetic_dataset(n_users=300, n_items=400, rank=10, train_density=0.08, test_density=0.02, noise_std=0.1, seed=123):
    rng = np.random.default_rng(seed)
    U0 = rng.standard_normal((n_users, rank))
    V0 = rng.standard_normal((n_items, rank))
    X = U0 @ V0.T + noise_std * rng.standard_normal((n_users, n_items))
    X = 3 + (X - X.mean()) / (X.std() + 1e-8)
    X = np.clip(X, 1.0, 5.0)
    mask_train = rng.random((n_users, n_items)) < train_density
    mask_remaining = ~mask_train
    mask_test = np.zeros_like(mask_train, dtype=bool)
    test_choices = np.where(mask_remaining)
    n_test = int(test_density * n_users * n_items)
    if len(test_choices[0]) > 0 and n_test > 0:
        pick = rng.choice(len(test_choices[0]), size=n_test, replace=False)
        mask_test[test_choices[0][pick], test_choices[1][pick]] = True
    return X.astype(np.float32), mask_train, mask_test

def load_ml100k(path: str):
    # Expects MovieLens 100k u.data (tab-separated: user item rating timestamp)
    data = np.loadtxt(path, dtype=int)
    u = data[:,0] - 1
    i = data[:,1] - 1
    r = data[:,2].astype(float)
    n_users = int(u.max()+1); n_items = int(i.max()+1)
    X = np.zeros((n_users, n_items), dtype=np.float32)
    X[:] = np.nan
    X[u, i] = r
    # create masks: 80% train of observed, 20% test
    obs = np.where(~np.isnan(X))
    m = len(obs[0])
    rng = np.random.default_rng(42)
    idx = rng.permutation(m)
    split = int(0.8 * m)
    train_pairs = (obs[0][idx[:split]], obs[1][idx[:split]])
    test_pairs = (obs[0][idx[split:]], obs[1][idx[split:]])
    mask_train = np.zeros_like(X, dtype=bool)
    mask_test = np.zeros_like(X, dtype=bool)
    mask_train[train_pairs] = True
    mask_test[test_pairs] = True
    # Fill missing with zeros for training routine; we only index by mask anyway
    X_filled = np.nan_to_num(X, nan=0.0).astype(np.float32)
    return X_filled, mask_train, mask_test

def add_gaussian_noise(X, sigma, seed=0):
    rng = np.random.default_rng(seed)
    return X + sigma * rng.standard_normal(X.shape).astype(X.dtype)

def try_import_user_methods(module_path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("user_adv", module_path)
    if spec is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print("WARNING: failed to import your module due to:", repr(e))
        return None
    fns = {
        "module": mod,
        "optimize_perturbation": getattr(mod, "optimize_perturbation", None),
        "optimize_perturbation_simplifed": getattr(mod, "optimize_perturbation_simplifed", None),
        "optimize_perturbation_comparison": getattr(mod, "optimize_perturbation_comparison", None),
        "OptimizationConfig": getattr(mod, "OptimizationConfig", None),
    }
    return fns


def build_attr_like(user_fns, variant: str, config_kwargs=None):
    import types
    config_kwargs = dict(config_kwargs or {})

    if user_fns is None or user_fns.get(variant) is None:
        def _fallback(X):
            raise RuntimeError(f"{variant} not available. Ensure deps installed and functions exist.")
        return _fallback

    # -------- helper: 规范化 TensorTrain 返回的 cores --------
    def _extract_cores(tt_obj):
        # 兼容 tensorly 的不同版本：可能是 .factors、.cores 或 tuple/list
        for attr in ("factors", "cores"):
            if hasattr(tt_obj, attr):
                return list(getattr(tt_obj, attr))
        if isinstance(tt_obj, (list, tuple)):
            return list(tt_obj)
        # 最后兜底：当成单个对象返回
        return [tt_obj]

    # -------- helper: 规范化函数返回值（E 或 X_adv 或 tuple）--------
    def _pick_tensor_same_shape(as_tuple_like, shape):
        # 从 tuple/list 里挑出与 X 同形状的 Tensor
        for obj in as_tuple_like:
            try:
                if hasattr(obj, "shape") and tuple(obj.shape) == tuple(shape):
                    return obj
            except Exception:
                pass
        return None

    def _ensure_global_hparams(mod, kv):
        # 部分实现可能直接使用全局变量名（如 inner_num_steps）
        for k, v in kv.items():
            if not hasattr(mod, k):
                try:
                    setattr(mod, k, v)
                except Exception:
                    pass

    def _build_config_obj(OptimizationConfig, cfg_kv):
        if OptimizationConfig is not None:
            import inspect
            sig = inspect.signature(OptimizationConfig)
            allowed = {p.name for p in sig.parameters.values()}
            filtered = {k: v for k, v in cfg_kv.items() if k in allowed}
            try:
                return OptimizationConfig(**filtered)
            except Exception:
                pass
        # 没有类或构造失败时，兜底成带属性的对象
        return SimpleNamespace(**cfg_kv)

    from types import SimpleNamespace

    def _perturb(X):
        import torch, tensorly as tl
        from tensorly.decomposition import tensor_train
        tl.set_backend("pytorch")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        X0 = X_t.clone()  # (a) 记录调用前的副本

        # Decompose X into tensor ring format
        ranks = [5, 5, 5]  # Rank for each factor tensor
        ranks = ranks + [ranks[0]]  # TR 的最后一个 rank 应该等于第一个 rank
        shape = list(X_t.shape)
        print(X_t.shape) 
        G_list = tr.initialize_tr_cores(shape, ranks, device="cuda")  # 初始化 TR 核心张量

        # # 将 tr_cores 转换为 NumPy 数组
        # tr_cores_np = [core.cpu().numpy() for core in tr_cores]

        # # 创建 G_list
        # G_list = tr_cores_np  # G_list 是 TR 核心张量的列表

        # --- 配置对象：始终是“有属性”的对象（OptimizationConfig 或 SimpleNamespace）---
        defaults = {
            "epsilon":          config_kwargs.get("epsilon", 1.0),
            "lr":               config_kwargs.get("lr", 1e-2),
            "outer_num_steps":  config_kwargs.get("outer_num_steps", 20),
            "inner_num_steps":  config_kwargs.get("inner_num_steps", 100),
            "outer_dis_num":    config_kwargs.get("outer_dis_num", 3),
            "inner_dis_num":    config_kwargs.get("inner_dis_num", 3),
            "inner_tol":        config_kwargs.get("inner_tol", 1e-5),
        }
        # 注入到模块全局，兼容直接读全局名
        for k, v in defaults.items():
            if not hasattr(user_fns["module"], k):
                setattr(user_fns["module"], k, v)

        OptimizationConfig = user_fns.get("OptimizationConfig")
        def _build_cfg(OptCfg, kv):
            if OptCfg is not None:
                import inspect
                sig = inspect.signature(OptCfg)
                filt = {k: v for k, v in kv.items() if k in {p.name for p in sig.parameters.values()}}
                try:
                    return OptCfg(**filt)
                except Exception:
                    pass
            return SimpleNamespace(**kv)

        cfg_obj = _build_cfg(OptimizationConfig, defaults)

        # --- 调用你的函数 ---
        fn = user_fns[variant]
        out = fn(X_t, G_list, cfg_obj)

        # (b) 若返回 None，视为就地修改。若确实发生改变，直接返回修改后的 X；否则给出清晰错误
        if out is None:
            if not torch.allclose(X_t, X0):
                return X_t.detach().cpu().numpy()
            raise RuntimeError(f"{variant} returned None and did not modify input tensor in-place.")

        # --- 非 None：可能是 E、X_adv 或 tuple/list ---
        def _pick_same_shape(obj, shape):
            if isinstance(obj, (list, tuple)):
                for o in obj:
                    if hasattr(o, "shape") and tuple(o.shape) == tuple(shape):
                        return o
                return None
            return obj

        cand = _pick_same_shape(out, X_t.shape)
        if cand is None or not hasattr(cand, "to"):
            raise TypeError(f"{variant} returned unsupported structure: {type(out)}")

        cand = cand.to(device=device, dtype=X_t.dtype)
        try:
            X_adv_t = X_t + cand  # 当作 E
        except Exception:
            X_adv_t = cand        # 当作 X_adv

        if torch.allclose(X_adv_t, X0):
            raise RuntimeError(f"{variant} produced no change (X_adv == X).")

        return X_adv_t.detach().cpu().numpy()

    return _perturb


def run_condition(name, X_true, mask_train, mask_test, perturbator, mf_kwargs=None):
    mf_kwargs = mf_kwargs or {}
    Xp_raw = perturbator(X_true)
    if Xp_raw is None:
        raise RuntimeError(
            f"{name}: perturbator returned None; ensure your function returns E or X_adv, "
            f"or let the adapter handle in-place by actually mutating the input tensor."
        )
    Xp = Xp_raw.astype(np.float32)

    X_train_obs = np.where(mask_train, Xp, 0.0)
    U, V = train_mf_sgd(X_train_obs, mask_train, **mf_kwargs)
    P = predict_all(U, V)
    return {"RMSE@test": rmse(P, X_true, mask_test), "MAE@test": mae(P, X_true, mask_test)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "ml100k"], default="synthetic")
    ap.add_argument("--ml100k_path", type=str, default="")
    ap.add_argument("--user_module", type=str, default="AT_TD/AT_TR_plus.py")
    ap.add_argument("--gauss_sigma", type=float, default=0.2)
    ap.add_argument("--mf_rank", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--epsilon", type=float, default=100.0)
    args = ap.parse_args()

    if args.data == "synthetic":
        X_true, mask_train, mask_test = make_synthetic_dataset()
    else:
        if not args.ml100k_path:
            raise ValueError("Provide --ml100k_path pointing to MovieLens 100k u.data")
        X_true, mask_train, mask_test = load_ml100k(args.ml100k_path)

    user_fns = try_import_user_methods(args.user_module)

    # Define perturbations
    clean = lambda X: X
    gaussian = lambda X: add_gaussian_noise(X, args.gauss_sigma, seed=0)
    attr     = build_attr_like(user_fns, "optimize_perturbation_comparison", {"epsilon": args.epsilon})
    fagadatr = build_attr_like(user_fns, "optimize_perturbation_simplifed", {"epsilon": args.epsilon})
    adatr    = build_attr_like(user_fns, "optimize_perturbation", {"epsilon": args.epsilon})

    mf_kwargs = dict(n_factors=args.mf_rank, n_epochs=args.epochs)

    results = []
    for name, pfun in [
        ("Clean", clean),
        ("Gaussian", gaussian),
        ("ATTR (comparison)", attr),
        ("FAG-AdaTR (simplifed)", fagadatr),
        ("AdaTR", adatr),
    ]:
        try:
            metrics = run_condition(name, X_true, mask_train, mask_test, pfun, mf_kwargs)
        except Exception as e:
            metrics = {"RMSE@test": np.nan, "MAE@test": np.nan, "ERROR": str(e)}
        row = {"Variant": name}; row.update(metrics)
        results.append(row)

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
