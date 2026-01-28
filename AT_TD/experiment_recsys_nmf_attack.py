
"""
End-to-end NMF robustness experiment comparing 4 variants:
1) Clean (no noise)
2) Gaussian noise
3) ATNMF attack (objective_function_ATNMF)
4) ANMF attack (objective_function_ANMF)

Usage:
  # Synthetic low-rank data
  python experiment_recsys_nmf_attack.py --data synthetic

  # MovieLens 100K (u.data, tab-separated: user item rating timestamp)
  python experiment_recsys_nmf_attack.py --data ml100k --ml100k_path /path/to/u.data

Optional args (see argparse):
  --user_module AT_TR_plus.py                # or your 442ba20d-....py
  --rank 40 --nmf_iters 150 --reg 1e-2
  --epsilon 0.5 --pgd_steps 40 --pgd_alpha 0.05 --pgd_norm linf
  --gauss_sigma 0.2
"""

import argparse
import math
import os   # 👈 补上这一行
import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
from types import SimpleNamespace
from typing import Optional, Tuple, Dict, Callable

USE_TORCH_ATTACKS = False


# ------------------------------------
# Utilities
# ------------------------------------
def rmse(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    diff = (pred - truth)[mask]
    return float(np.sqrt(np.mean(diff * diff))) if np.any(mask) else float("nan")

def mae(pred: np.ndarray, truth: np.ndarray, mask: np.ndarray) -> float:
    diff = np.abs((pred - truth)[mask])
    return float(np.mean(diff)) if np.any(mask) else float("nan")

def add_gaussian_noise(X: np.ndarray, sigma: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return X + sigma * rng.standard_normal(X.shape).astype(X.dtype)

# ------------------------------------
# Data
# ------------------------------------
def make_synthetic_dataset(
    n_users: int = 300,
    n_items: int = 400,
    rank: int = 10,
    train_density: float = 0.08,
    test_density: float = 0.02,
    noise_std: float = 0.1,
    seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

def load_ml100k(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, dtype=int)
    u = data[:,0] - 1
    i = data[:,1] - 1
    r = data[:,2].astype(float)
    n_users = int(u.max()+1); n_items = int(i.max()+1)
    X = np.full((n_users, n_items), np.nan, dtype=np.float32)
    X[u, i] = r
    obs = np.where(~np.isnan(X))
    m = len(obs[0])
    rng = np.random.default_rng(42)
    idx = rng.permutation(m)
    split = int(0.8 * m)
    train_pairs = (obs[0][idx[:split]], obs[1][idx[:split]])
    test_pairs  = (obs[0][idx[split:]], obs[1][idx[split:]])
    mask_train = np.zeros_like(X, dtype=bool)
    mask_test  = np.zeros_like(X, dtype=bool)
    mask_train[train_pairs] = True
    mask_test[test_pairs]   = True
    X_filled = np.nan_to_num(X, nan=0.0).astype(np.float32)
    return X_filled, mask_train, mask_test

# ------------------------------------
# Masked NMF with multiplicative updates (squared loss + L2 reg)
# ------------------------------------
def nmf_mu_masked(
    X_obs: np.ndarray,  # observed entries at train mask, others zero
    M: np.ndarray,      # boolean mask for observed train entries
    rank: int = 40,
    n_iters: int = 200,
    reg: float = 1e-2,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimize 0.5 * || M ◦ (X_obs - W H^T) ||_F^2 + (reg/2)(||W||_F^2 + ||H||_F^2)
    using multiplicative updates adapted to the masked case.
    Ensures W,H >= 0.
    """
    rng = np.random.default_rng(seed)
    n_users, n_items = X_obs.shape
    # non-negative init
    W = np.maximum(1e-3, rng.random((n_users, rank)).astype(np.float32))
    H = np.maximum(1e-3, rng.random((n_items, rank)).astype(np.float32))
    M_f = M.astype(np.float32)
    eps = 1e-8

    for _ in range(n_iters):
        # Compute WH and masked versions
        WH = W @ H.T
        # Update W
        numer_W = (M_f * X_obs) @ H
        denom_W = (M_f * WH) @ H + reg * W + eps
        W *= numer_W / denom_W
        # Update H
        WH = W @ H.T  # refresh
        numer_H = (M_f * X_obs).T @ W
        denom_H = ((M_f * WH).T) @ W + reg * H + eps
        H *= numer_H / denom_H

        # Avoid numerical issues
        W = np.maximum(W, 0.0)
        H = np.maximum(H, 0.0)

    return W, H

def predict_all_nmf(W: np.ndarray, H: np.ndarray) -> np.ndarray:
    return W @ H.T

# ------------------------------------
# User module import + objective wrapping
# ------------------------------------
def try_import_user_module(module_path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("user_nmf", module_path)
    if spec is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        print("WARNING: failed to import user module:", repr(e))
        return None
    return mod

def build_cfg_like(obj: Optional[object], defaults: Dict) -> object:
    if obj is not None and callable(obj):
        try:
            return obj(**defaults)
        except Exception:
            pass
    # fallback
    return SimpleNamespace(**defaults)

def build_user_objective(mod, name: str, default_loss):
    """
    Return a callable loss(X_t, W_t, H_t, M_t, cfg) -> scalar torch tensor.
    If user objective not found or fails, fallback to default_loss.
    """
    def _default(X_t, W_t, H_t, M_t, cfg):
        # 0.5 * || M ◦ (X - W H^T) ||^2
        return 0.5 * ((M_t * (X_t - W_t @ H_t.T))**2).sum()
    if default_loss is None:
        default_loss = _default

    if mod is None:
        return default_loss

    fn = getattr(mod, name, None)
    if fn is None:
        return default_loss

    # Try a few common signatures
    def _wrapped(X_t, W_t, H_t, M_t, cfg):
        for args in [
            (X_t, W_t, H_t, M_t, cfg),
            (X_t, W_t, H_t, cfg),
            (X_t, M_t, cfg),
            (X_t, cfg),
            (X_t,),
        ]:
            try:
                out = fn(*args)
                if hasattr(out, "shape"):  # tensor-like scalar
                    return out
                # If returns python float
                import torch
                return torch.as_tensor(out, dtype=X_t.dtype, device=X_t.device)
            except TypeError:
                continue
        # Fallback
        return default_loss(X_t, W_t, H_t, M_t, cfg)
    return _wrapped

# ------------------------------------
# PGD attack on X with user objective (torch autograd if available)
# ------------------------------------
def pgd_attack_nmf(
    X: np.ndarray,          # clean full matrix
    M_train: np.ndarray,    # boolean mask for observed training entries
    rank: int,
    steps: int,
    alpha: float,
    epsilon: float,
    pnorm: str,
    rating_min: float,
    rating_max: float,
    user_objective: Optional[Callable],
    nmf_iters_inner: int = 50,
    reg: float = 1e-2,
    seed: int = 0,
) -> np.ndarray:
    """
    Perform projected gradient ascent on X to maximize user_objective or masked MSE,
    while keeping W,H fixed to the current NMF factors (re-estimated once before PGD).
    """
    try:
        import torch
    except Exception:
        torch = None

    if not USE_TORCH_ATTACKS:
        torch = None

    # 1) Fit NMF on current (clean) observed train entries (constants in attack loop)
    X_obs = np.where(M_train, X, 0.0).astype(np.float32)
    W, H = nmf_mu_masked(X_obs, M_train, rank=rank, n_iters=nmf_iters_inner, reg=reg, seed=seed)

    # 2) Torch tensors for autograd (if torch available)
    if torch is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        X0 = torch.tensor(X, dtype=torch.float32, device=device)
        X_adv = X0.clone().detach().requires_grad_(True)
        W_t = torch.tensor(W, dtype=torch.float32, device=device)
        H_t = torch.tensor(H, dtype=torch.float32, device=device)
        M_t = torch.tensor(M_train.astype(np.float32), dtype=torch.float32, device=device)

        # Loss function
        if user_objective is None:
            def loss_fn(Xt):  # default masked MSE
                return 0.5 * ((M_t * (Xt - W_t @ H_t.T))**2).sum()
        else:
            def loss_fn(Xt):
                return user_objective(Xt, W_t, H_t, M_t, SimpleNamespace(epsilon=epsilon, lr=alpha))

        # 3) PGD steps
        for _ in range(steps):
            if X_adv.grad is not None:
                X_adv.grad.zero_()
            loss = loss_fn(X_adv)
            loss.backward()
            g = X_adv.grad

            if pnorm == "linf":
                X_adv = X_adv.detach() + alpha * g.sign()
                X_adv = torch.max(torch.min(X_adv, X0 + epsilon), X0 - epsilon)
            elif pnorm == "l2":
                g_flat = g.view(-1)
                norm = torch.norm(g_flat) + 1e-12
                step = alpha * g / norm
                X_adv = X_adv.detach() + step
                # project to l2 ball
                d = (X_adv - X0).view(-1)
                dn = torch.norm(d) + 1e-12
                if dn > epsilon:
                    X_adv = X0 + (d / dn) * epsilon
            else:
                raise ValueError("pnorm must be 'linf' or 'l2'")

            # clip to rating range
            X_adv = torch.clamp(X_adv, rating_min, rating_max).detach().requires_grad_(True)

        return X_adv.detach().cpu().numpy().astype(np.float32)

    # ----- Fallback: no torch; use numpy gradient of masked MSE with fixed W,H -----
    X0 = X.copy()
    X_adv = X.copy()
    for _ in range(steps):
        # gradient of 0.5 * || M ◦ (X - W H^T) ||^2 wrt X is M ◦ (X - W H^T)
        grad = np.where(M_train, X_adv - (W @ H.T), 0.0).astype(np.float32)

        if pnorm == "linf":
            X_adv = X_adv + alpha * np.sign(grad)
            X_adv = np.minimum(np.maximum(X_adv, X0 - epsilon), X0 + epsilon)
        elif pnorm == "l2":
            norm = np.linalg.norm(grad.ravel()) + 1e-12
            step = alpha * grad / norm
            X_adv = X_adv + step
            d = X_adv - X0
            dn = np.linalg.norm(d.ravel()) + 1e-12
            if dn > epsilon:
                X_adv = X0 + d * (epsilon / dn)
        else:
            raise ValueError("pnorm must be 'linf' or 'l2'")

        X_adv = np.clip(X_adv, rating_min, rating_max)

    return X_adv.astype(np.float32)

# ------------------------------------
# Experiment harness
# ------------------------------------
def run_condition_nmf(
    name: str,
    X_true: np.ndarray,
    mask_train: np.ndarray,
    mask_test: np.ndarray,
    perturbator: Callable[[np.ndarray], np.ndarray],
    rank: int,
    nmf_iters: int,
    reg: float,
    rating_min: float,
    rating_max: float,
    norm_label: str = "N/A",
) -> Dict[str, float]:
    Xp = perturbator(X_true)
    if Xp is None:
        raise RuntimeError(f"{name}: perturbator returned None")
    Xp = Xp.astype(np.float32)

    # Only reveal (possibly perturbed) TRAIN entries to NMF
    X_train_obs = np.where(mask_train, Xp, 0.0)

    # Train NMF on observed train entries
    W, H = nmf_mu_masked(X_train_obs, mask_train, rank=rank, n_iters=nmf_iters, reg=reg, seed=0)
    P = predict_all_nmf(W, H)

    # Evaluate against CLEAN ground-truth on test entries
    perturb = Xp - X_true
    pert_f = float(np.linalg.norm(perturb)) if np.any(perturb) else 0.0
    pert_inf = float(np.max(np.abs(perturb))) if np.any(perturb) else 0.0
    return {
        "Variant": name,
        "Norm": norm_label,
        "RMSE@test": rmse(P, X_true, mask_test),
        "MAE@test": mae(P, X_true, mask_test),
        "||E||_F": pert_f,
        "||E||_inf": pert_inf,
    }

# ------------------------------------
# Main
# ------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "ml100k"], default="synthetic")
    ap.add_argument("--ml100k_path", type=str, default="")
    ap.add_argument("--rank", type=int, default=40)
    ap.add_argument("--nmf_iters", type=int, default=200)
    ap.add_argument("--reg", type=float, default=1e-2)
    ap.add_argument("--rating_min", type=float, default=1.0)
    ap.add_argument("--rating_max", type=float, default=5.0)

    # Gaussian
    ap.add_argument(
        "--gauss_sigma",
        type=float,
        default=None,
        help="Stddev for Gaussian noise; defaults to epsilon/√N to match Frobenius energy.",
    )

    # Attack hyperparams
    ap.add_argument("--epsilon", type=float, default=0.5)
    ap.add_argument("--pgd_steps", type=int, default=100)
    ap.add_argument("--pgd_alpha", type=float, default=0.05)
    ap.add_argument("--pgd_norm", choices=["linf", "l2"], default="l2")
    ap.add_argument("--nmf_inner_for_attack", type=int, default=50)
    ap.add_argument(
        "--compare-linf-attacks",
        action="store_true",
        help="In addition to the base norm, also evaluate ATNMF/AdaNMF under L-infinity constraints.",
    )

    # User module (optional)
    ap.add_argument("--user_module", type=str, default="")

    args = ap.parse_args()

    # Load data
    if args.data == "synthetic":
        X_true, mask_train, mask_test = make_synthetic_dataset()
    else:
        if not args.ml100k_path:
            raise ValueError("Provide --ml100k_path pointing to MovieLens 100k u.data")
        X_true, mask_train, mask_test = load_ml100k(args.ml100k_path)

    # Import user module if provided
    mod = None
    if args.user_module and os.path.exists(args.user_module):
        mod = try_import_user_module(args.user_module)

    # Build user objectives
    user_obj_AT = build_user_objective(mod, "objective_function_ATNMF", default_loss=None)
    user_obj_AN = build_user_objective(mod, "objective_function_ANMF", default_loss=None)

    total_entries = X_true.size if X_true.size > 0 else 1
    epsilon_f = float(args.epsilon)
    epsilon_linf = float(args.epsilon) / math.sqrt(total_entries)
    gaussian_sigma = (
        float(args.gauss_sigma)
        if args.gauss_sigma is not None
        else (epsilon_f / math.sqrt(total_entries))
    )
    print(f"Gaussian sigma adjusted to {gaussian_sigma:.6f} to match energy budget.")

    # Perturbators
    rating_min, rating_max = args.rating_min, args.rating_max

    def clean(X): 
        return X

    def gauss(X):
        Xg = add_gaussian_noise(X, gaussian_sigma, seed=0)
        diff = Xg - X
        norm = np.linalg.norm(diff.ravel()) + 1e-12
        if norm > epsilon_f:
            diff = diff * (epsilon_f / norm)
            Xg = X + diff.reshape(X.shape)
        return np.clip(Xg, rating_min, rating_max)

    def make_attack_fn(user_objective, pnorm_value, epsilon_value):
        def attack_fn(X):
            return pgd_attack_nmf(
                X, mask_train, rank=args.rank, steps=args.pgd_steps, alpha=args.pgd_alpha,
                epsilon=epsilon_value, pnorm=pnorm_value, rating_min=rating_min, rating_max=rating_max,
                user_objective=user_objective, nmf_iters_inner=args.nmf_inner_for_attack, reg=args.reg, seed=0
            )
        return attack_fn

    # Run
    results = []
    base_conditions = [("Clean", clean), ("Gaussian", gauss)]
    for name, pfun in base_conditions:
        try:
            metrics = run_condition_nmf(
                name, X_true, mask_train, mask_test, pfun,
                rank=args.rank, nmf_iters=args.nmf_iters, reg=args.reg,
                rating_min=rating_min, rating_max=rating_max,
                norm_label="N/A",
            )
        except Exception as e:
            metrics = {
                "Variant": name,
                "Norm": "N/A",
                "RMSE@test": np.nan,
                "MAE@test": np.nan,
                "||E||_F": np.nan,
                "||E||_inf": np.nan,
                "ERROR": str(e),
            }
        results.append(metrics)

    def norm_label_and_eps(pnorm_value: str) -> Tuple[str, float]:
        if pnorm_value == "linf":
            return "L∞", epsilon_linf
        else:
            return "F", epsilon_f

    requested_norms = [args.pgd_norm]
    if args.compare_linf_attacks and "linf" not in requested_norms:
        requested_norms.append("linf")

    for pnorm_value in requested_norms:
        norm_label, epsilon_value = norm_label_and_eps(pnorm_value)
        atnmf = make_attack_fn(user_obj_AT, pnorm_value, epsilon_value)
        anmf = make_attack_fn(user_obj_AN, pnorm_value, epsilon_value)
        for base_name, pfun in [("ATNMF", atnmf), ("ANMF", anmf)]:
            variant_name = f"{base_name} ({norm_label})"
            try:
                metrics = run_condition_nmf(
                    variant_name, X_true, mask_train, mask_test, pfun,
                    rank=args.rank, nmf_iters=args.nmf_iters, reg=args.reg,
                    rating_min=rating_min, rating_max=rating_max,
                    norm_label=norm_label,
                )
            except Exception as e:
                metrics = {
                    "Variant": variant_name,
                    "Norm": norm_label,
                    "RMSE@test": np.nan,
                    "MAE@test": np.nan,
                    "||E||_F": np.nan,
                    "||E||_inf": np.nan,
                    "ERROR": str(e),
                }
            results.append(metrics)

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
