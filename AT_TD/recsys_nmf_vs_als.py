
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NMF vs ALS (masked) on implicit/explicit feedback with train/test masks.

- Weighted-NMF (multiplicative updates) that respects the observation mask W.
- ALS (ridge-regularized) that also respects W.
- Metrics: RMSE / MAE computed on held-out test entries only.

Usage (example):
    python recsys_nmf_vs_als.py --rank 40 --nmf-iters 300 --als-iters 30 --reg 0.1 --seed 42

You can replace the synthetic data generator with your dataset loader:
    - Provide R (n_users x n_items) with *missing entries set to 0*
    - Provide boolean W_obs mask (same shape) with True at observed entries
"""
import argparse
import numpy as np

def set_seed(seed: int):
    return np.random.default_rng(seed)

def train_test_split_mask(W_obs: np.ndarray, test_per_user: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Split each user's observed entries into train/test masks (disjoint)."""
    n_users, n_items = W_obs.shape
    W_train = np.zeros_like(W_obs, dtype=bool)
    W_test  = np.zeros_like(W_obs, dtype=bool)
    for u in range(n_users):
        items_u = np.flatnonzero(W_obs[u])
        if items_u.size == 0:
            continue
        rng.shuffle(items_u)
        t = min(test_per_user, items_u.size)
        test_idx = items_u[:t]
        train_idx = items_u[t:]
        W_train[u, train_idx] = True
        W_test[u,  test_idx]  = True
    return W_train, W_test

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if y_true.size else float('nan')

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred))) if y_true.size else float('nan')

def als_matrix_factorization(R: np.ndarray, W: np.ndarray, rank=20, n_iters=30, reg=0.1, seed=42):
    """
    ALS with mask W (True=observed). Minimizes sum_{(u,i) in Obs} (R_ui - U_u V_i^T)^2 + reg(||U||_F^2 + ||V||_F^2)
    R: (n_users x n_items), zeros at missing locations are ignored via W.
    """
    rng = set_seed(seed)
    n_users, n_items = R.shape
    U = 0.01 * rng.standard_normal((n_users, rank))
    V = 0.01 * rng.standard_normal((n_items, rank))
    I = np.eye(rank)

    obs_items_per_user = [np.flatnonzero(W[u]) for u in range(n_users)]
    obs_users_per_item = [np.flatnonzero(W[:, i]) for i in range(n_items)]

    for _ in range(n_iters):
        # update U
        for u in range(n_users):
            idx = obs_items_per_user[u]
            if idx.size == 0:
                continue
            V_u = V[idx]                      # (m x r)
            r_u = R[u, idx]                   # (m,)
            A = V_u.T @ V_u + reg * I         # (r x r)
            b = V_u.T @ r_u                   # (r,)
            U[u] = np.linalg.solve(A, b)
        # update V
        for i in range(n_items):
            idx = obs_users_per_item[i]
            if idx.size == 0:
                continue
            U_i = U[idx]                      # (m x r)
            r_i = R[idx, i]                   # (m,)
            A = U_i.T @ U_i + reg * I         # (r x r)
            b = U_i.T @ r_i                   # (r,)
            V[i] = np.linalg.solve(A, b)
    return U, V

def weighted_nmf(R: np.ndarray, W: np.ndarray, rank=20, n_iters=300, reg=1e-2, seed=42, eps=1e-10):
    """
    Weighted NMF with mask W via multiplicative updates (β=2, squared loss):
        min_{U,V>=0}  0.5 * || W ⊙ (R - U V^T) ||_F^2 + (reg/2)(||U||_F^2 + ||V||_F^2)

    Updates (all operations elementwise where appropriate):
        X = W ⊙ (U V^T)
        U <- U ⊙ ( (W ⊙ R) V ) / ( (W ⊙ X) V + reg U + eps )
        V <- V ⊙ ( (W ⊙ R)^T U ) / ( (W ⊙ X)^T U + reg V + eps )

    Notes:
    - Requires R >= 0 (ratings must be non-negative). If your ratings include negatives or are mean-centered,
      add a constant shift and remember to subtract it back when evaluating.
    - Zeros in R should mean "missing or zero rating"; only entries with W=True contribute to the loss.
    """
    assert R.shape == W.shape
    if np.any(R < 0):
        raise ValueError("R has negative values; NMF assumes non-negative data. Consider shifting R >= 0 first.")

    rng = set_seed(seed)
    n_users, n_items = R.shape
    U = np.abs(0.1 * rng.standard_normal((n_users, rank)))
    V = np.abs(0.1 * rng.standard_normal((n_items, rank)))

    for _ in range(n_iters):
        X = U @ V.T                     # reconstruction
        WX = W * X
        WR = W * R

        # U update
        numU = WR @ V                   # (n_users x rank)
        denU = WX @ V + reg * U + eps
        U *= numU / np.maximum(denU, eps)

        # V update
        X = U @ V.T                     # recompute after U update
        WX = W * X
        numV = WR.T @ U                 # (n_items x rank)
        denV = WX.T @ U + reg * V + eps
        V *= numV / np.maximum(denV, eps)

    return U, V

def evaluate(R: np.ndarray, U: np.ndarray, V: np.ndarray, W_eval: np.ndarray) -> dict:
    P = U @ V.T
    y_true = R[W_eval]
    y_pred = P[W_eval]
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred)}

def synthetic_dataset(n_users=500, n_items=800, rank=10, density=0.05, noise=0.1, seed=42):
    """
    Generate a non-negative low-rank ground-truth and sparse observations.
    Ratings are clipped to [0, 5]. Missing entries are zero in R but W_obs marks observed ones.
    """
    rng = set_seed(seed)
    U0 = np.abs(rng.standard_normal((n_users, rank)))
    V0 = np.abs(rng.standard_normal((n_items, rank)))
    full = U0 @ V0.T
    full = 5.0 * full / (full.max() + 1e-8)
    W_obs = rng.random((n_users, n_items)) < density
    R = np.zeros_like(full)
    R[W_obs] = np.clip(full[W_obs] + noise * rng.standard_normal(W_obs.sum()), 0.0, 5.0)
    return R, W_obs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=40)
    ap.add_argument("--nmf-iters", type=int, default=300)
    ap.add_argument("--als-iters", type=int, default=30)
    ap.add_argument("--reg", type=float, default=0.1)
    ap.add_argument("--test-per-user", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-synth", action="store_true", help="Run on a synthetic dataset")
    args = ap.parse_args()

    rng = set_seed(args.seed)

    # === Replace this block with your own loader ===
    if args.use_synth:
        R, W_obs = synthetic_dataset(seed=args.seed)
    else:
        # Example placeholder: expect you to fill R, W_obs from your data.
        # R must be (n_users x n_items), non-negative; W_obs is boolean mask where R is observed.
        raise SystemExit("Please load your dataset: set R (>=0) and W_obs (bool mask), or run with --use_synth")
    # ===============================================

    # Split observed entries into train/test
    W_train, W_test = train_test_split_mask(W_obs, test_per_user=args.test_per_user, rng=rng)

    # Train ALS (on train mask)
    U_als, V_als = als_matrix_factorization(R, W_train, rank=args.rank, n_iters=args.als_iters, reg=args.reg, seed=args.seed)
    metrics_als = evaluate(R, U_als, V_als, W_test)

    # Train Weighted-NMF (on train mask)
    U_nmf, V_nmf = weighted_nmf(R, W_train, rank=args.rank, n_iters=args.nmf_iters, reg=args.reg, seed=args.seed)
    metrics_nmf = evaluate(R, U_nmf, V_nmf, W_test)

    print("=== Test metrics (held-out) ===")
    print(f"ALS : RMSE={metrics_als['rmse']:.4f}, MAE={metrics_als['mae']:.4f}")
    print(f"NMF : RMSE={metrics_nmf['rmse']:.4f}, MAE={metrics_nmf['mae']:.4f}")

if __name__ == "__main__":
    main()
