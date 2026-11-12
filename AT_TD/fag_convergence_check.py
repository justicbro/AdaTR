"""Measure the quantities of the revised Theorems 5.5/5.6 for FAG-AdaTR (paper/FAG_theory_revision).

Algorithm 2 is run exactly as attack_methods.optimize_perturbation_simplifed does it,
    E <- Pi_B(E + eta * sum_n w_n Fold_n(E_[n] P_n)),   P_n = pinv(G_{!=n}) G_{!=n},   w_n = I_n / sum_j I_j,
where eta is the step that really multiplies the gradient (the code uses 10 * config.lr). Per outer step t we record
    delta[t] = ||E^(t+1) - E^(t)||_F
    rho[t]   = sum_n w_n ||P_n^(t+1) - P_n^(t)||_2   (sine of the largest principal angle between the row spaces)
    q_t(E^(t)), q_t(E^(t+1))                          (checks of item 1 and of (C.1))
and at t* = argmin_t delta[t] the gradient mismatch of Theorem 5.6,
    eps_g = ||grad g(E^(t*)) - grad g~_t*(E^(t*))||_F,   g = 1/2 ||X - TR(G^(T)(E))||_F^2,
with grad g from autodiff through the T_in ALS sweeps (the AdaTR gradient, checkpointed per sweep to fit in memory).

Examples (from the project root):
    CUDA_VISIBLE_DEVICES=0 python AT_TD/fag_convergence_check.py --kind image --items 1 2 --eta 1.0
    CUDA_VISIBLE_DEVICES=1 python AT_TD/fag_convergence_check.py --kind video --items 1 2 3 --eta 1.0
    python AT_TD/fag_convergence_check.py --summarize AT_TD/fag_convergence_results
"""
import argparse
import glob
import os
import time

import numpy as np
import scipy.io
import tensorly as tl
import torch
from PIL import Image
from torch.utils.checkpoint import checkpoint

import TR_func as tr

tl.set_backend("pytorch")
HERE = os.path.dirname(os.path.abspath(__file__))
F32_EPS = torch.finfo(torch.float32).eps


# ---------------------------------------------------------------- data
def load_image(path, resize_width=None):
    image = Image.open(path).convert("RGB")
    if resize_width:  # proportional resize, as process_image(..., resize_width=...) in AT_AdaTR2.py
        height = max(1, int(round(image.size[1] * resize_width / image.size[0])))
        image = image.resize((resize_width, height), Image.BILINEAR)
    return torch.tensor(np.array(image) / 255.0, dtype=torch.float32)


def load_video(path, frames, size):
    data = scipy.io.loadmat(path)
    arr = next(v for v in data.values() if isinstance(v, np.ndarray) and v.ndim == 4).astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = arr[..., :frames]  # first frames, as AT_TR4video_exp.py
    if size is None:  # native size, as AT_TR4video_exp.py (video 1 stays 288x352)
        return torch.from_numpy(np.ascontiguousarray(arr))
    width, height = size
    if arr.shape[:2] != (height, width):  # same resize as load_video_tensor in AT_AdaTR2.py
        out = np.empty((height, width, arr.shape[2], arr.shape[3]), dtype=np.float32)
        for t in range(arr.shape[3]):
            frame = Image.fromarray(np.clip(arr[..., t] * 255.0, 0, 255).astype(np.uint8))
            out[..., t] = np.asarray(frame.resize((width, height), Image.BILINEAR), dtype=np.float32) / 255.0
        arr = out
    return torch.from_numpy(np.ascontiguousarray(arr))


# ---------------------------------------------------------------- unfoldings
def mode_perm(N, n):
    return [n] + list(range(n + 1, N)) + list(range(n))


def unfold(T, n):
    return tr.generalized_k_unfolding(T, mode_perm(T.dim(), n), 0)


def fold(M, n, shape):
    perm = mode_perm(len(shape), n)
    return M.reshape(*[shape[d] for d in perm]).permute(*[perm.index(i) for i in range(len(shape))])


def subchain(cores, n):
    X_hat = tr.contract_except_n(cores, n)
    dims = list(range(X_hat.dim()))
    return tr.generalized_k_unfolding(X_hat, [dims[-1]] + dims[:-1], 1)  # G_{!=n}, shape (R^2, prod_{j!=n} I_j)


def row_basis(G):
    """Orthonormal basis (float64) of the row space kept by torch.linalg.pinv(G) with its default cutoff."""
    G64 = G.double()
    lam, U = torch.linalg.eigh(G64 @ G64.T)
    s = lam.clamp_min(0).sqrt()
    keep = s > F32_EPS * max(G.shape) * s.max()
    return (G64.T @ U[:, keep]) / s[keep]


def proj_dist(V1, V2):
    """||V1 V1^T - V2 V2^T||_2 = sin(largest principal angle); 1 if the dimensions differ."""
    if V1.shape[1] != V2.shape[1]:
        return 1.0
    W = V2 - V1 @ (V1.T @ V2)
    return float(torch.linalg.eigvalsh(W.T @ W).max().clamp_min(0).sqrt())


def project_ball(E, eps):
    norm = torch.norm(E)
    return E * (eps ** 0.5 / norm) if norm ** 2 > eps else E


def q_value(E, bases, w):
    return sum(0.5 * w[n] * float(torch.norm(unfold(E, n).double() @ V) ** 2) for n, V in enumerate(bases))


# ---------------------------------------------------------------- TR-ALS with autodiff (AdaTR gradient)
def _sweep(Xp, *cores):
    cores = list(cores)
    for n in range(len(cores)):
        cores[n] = tr.als_update_tr(Xp, cores, n, "cuda")
    return tuple(cores)


def tr_decompose2_checkpointed(Xp, cores_ori, max_iter, tol):
    """Same forward pass as TR_func.tr_decompose2, with one checkpoint per ALS sweep."""
    cores = [c.clone() for c in cores_ori]
    prev = float("inf")
    for _ in range(max_iter):
        cores = list(checkpoint(_sweep, Xp, *cores, use_reentrant=False))
        with torch.no_grad():
            loss = float(torch.norm(Xp - tl.tr_to_tensor(cores)) / torch.norm(Xp))
        if abs(prev - loss) < tol:
            break
        prev = loss
    return cores


def g_value(X, E, cores0, args):
    with torch.no_grad():
        cores = tr.tr_decompose2(X + E, cores0, max_iter=args.inner, tol=args.inner_tol, dis_num=10 ** 9)
        return 0.5 * float(torch.norm(X - tl.tr_to_tensor(cores)) ** 2)


# ---------------------------------------------------------------- one tensor
def run_one(X, cores0, E0, eps, args):
    shape, N = list(X.shape), X.dim()
    w = [s / sum(shape) for s in shape]
    eta = args.eta
    E = E0.clone()
    delta, rho, recon, ranks = [], [], [], []
    q_now, q_next_same, step_err, ideal_err, dense_err = [], [], [], [], []
    E_hist, bases_prev = [], None

    def surrogate(E, cores):
        """bases of P_n, grad g~ = sum_n w_n Fold(E_[n] P_n), and the code's update before projection."""
        bases, gtilde, E_sum, err = [], torch.zeros_like(E), torch.zeros_like(E), 0.0
        E_cur = E  # "sequential": mode n starts from mode n-1's result (old main scripts overwrite E in the loop)
        for n in range(N):
            X_u, E_u, Gr = unfold(X, n), unfold(E, n), subchain(cores, n)
            E_in = unfold(E_cur, n) if args.update == "sequential" else E_u  # "jacobi": attack_methods.py, Eq. (19)
            pinv = torch.linalg.pinv(Gr)
            code_grad = (X_u - (X_u + E_in) @ pinv @ Gr) @ Gr.T @ pinv.T  # attack_methods.py line 199
            E_mode = fold(E_in - eta * code_grad, n, shape)
            E_sum += w[n] * E_mode
            if args.update == "sequential":
                E_cur = E_mode
            V = row_basis(Gr)
            EP = (E_u.double() @ V) @ V.T
            gtilde += w[n] * fold(EP.float(), n, shape)
            E_in_P = EP if E_in is E_u else (E_in.double() @ V) @ V.T
            err = max(err, float(torch.norm(-code_grad.double() - E_in_P) / torch.norm(E_in_P).clamp_min(1e-30)))
            bases.append(V)
        return bases, gtilde, E_sum, err

    with torch.no_grad():
        for t in range(args.outer + 1):
            cores = tr.tr_decompose2(X + E, cores0, max_iter=args.inner, tol=args.inner_tol, dis_num=10 ** 9)
            bases, gtilde, E_sum, err = surrogate(E, cores)
            if bases_prev is not None:
                rho.append(sum(w[n] * proj_dist(bases_prev[n], bases[n]) for n in range(N)))
                for n in range(N if t <= 5 else 0):  # spot check of proj_dist against dense projectors, small modes
                    if bases[n].shape[0] <= 6000 and bases[n].shape[1] == bases_prev[n].shape[1]:
                        D = bases[n] @ bases[n].T - bases_prev[n] @ bases_prev[n].T
                        dense = float(torch.linalg.eigvalsh(D).abs().max())
                        dense_err.append(abs(dense - proj_dist(bases_prev[n], bases[n])))
            q_now.append(q_value(E, bases, w))
            recon.append(float(torch.norm(X - tl.tr_to_tensor(cores))))  # recon[K] = attack result of E^(K)
            if t == args.outer or (t > 0 and delta[-1] / (torch.norm(E_hist[-1]).item() + 1e-12) < args.outer_tol):
                break  # the last pass only provides P^(K), q_K(E^(K)) and the final reconstruction error
            ranks.append([V.shape[1] for V in bases])
            step_err.append(err)
            E_new = project_ball(E_sum, eps)
            ideal_err.append(float(torch.norm(E_new - project_ball(E + eta * gtilde, eps)) / torch.norm(E_new - E)))
            delta.append(float(torch.norm(E_new - E)))
            q_next_same.append(q_value(E_new, bases, w))
            E_hist.append(E.cpu())
            E, bases_prev = E_new, bases

    K = len(delta)
    delta, rho, q_now, q_next_same = map(np.array, (delta, rho, q_now, q_next_same))
    out = dict(K=K, delta=delta, rho=rho, recon=np.array(recon), ranks=np.array(ranks), q_now=q_now,
               q_next_same=q_next_same, step_err=np.array(step_err), ideal_err=np.array(ideal_err),
               dense_err=np.array(dense_err) if dense_err else np.zeros(1),
               item1_slack=(q_next_same - q_now[:K] - delta ** 2 / eta) / eps,
               c1_slack=(q_now[1:K + 1] - q_now[:K] - delta ** 2 / eta + 0.5 * eps * rho) / eps)

    def grad_check(t):
        """eps_g at E^(t): autodiff grad g vs grad g~ (Eq. 19) and vs the direction the code really used."""
        E_t = E_hist[t].cuda().requires_grad_(True)
        cores = tr_decompose2_checkpointed(X + E_t, cores0, args.inner, args.inner_tol)
        TN = tl.tr_to_tensor(cores)
        loss = 0.5 * torch.norm(X - TN) ** 2
        (grad_g,) = torch.autograd.grad(loss, E_t)
        with torch.no_grad():
            E_t = E_t.detach()
            _, gtilde, E_sum, _ = surrogate(E_t, [c.detach() for c in cores])
            direction = (E_sum - E_t) / eta  # equals grad g~ for "jacobi"
            G_eta = (project_ball(E_t + eta * grad_g, eps) - E_t) / eta
            # finite-difference check of the autodiff gradient along its own direction and along grad g~
            h = 1e-3 * eps ** 0.5
            fd = {}
            for name, d in (("grad_g", grad_g), ("grad_gtilde", gtilde)):
                d = d / torch.norm(d)
                fd[name] = ((g_value(X, E_t + h * d, cores0, args) - g_value(X, E_t - h * d, cores0, args)) / (2 * h),
                            float(torch.sum(grad_g * d)), float(torch.sum(gtilde * d)))
            cos = lambda a, b: float(torch.sum(a * b) / (torch.norm(a) * torch.norm(b)))
            return dict(eps_g=float(torch.norm(grad_g - gtilde)), grad_g_norm=float(torch.norm(grad_g)),
                        grad_gtilde_norm=float(torch.norm(gtilde)), cos=cos(grad_g, gtilde),
                        eps_g_dir=float(torch.norm(grad_g - direction)), dir_norm=float(torch.norm(direction)),
                        cos_dir=cos(grad_g, direction), G_eta_norm=float(torch.norm(G_eta)),
                        recon_ckpt=float(torch.norm(X - TN.detach())), recon_loop=out["recon"][t],
                        fd_grad_g=np.array(fd["grad_g"]), fd_grad_gtilde=np.array(fd["grad_gtilde"]))

    if args.grad_check:
        tstar = int(np.argmin(delta))
        at_tstar = grad_check(tstar)
        at_last = at_tstar if tstar == K - 1 else grad_check(K - 1)
        out.update(tstar=tstar, **at_tstar, **{f"last_{k}": v for k, v in at_last.items()})
    return out


def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    for item in args.items:
        idx = int(item)
        if args.kind == "image":
            X = load_image(os.path.join(args.image_dir, f"{idx}.png"), args.resize_width)
        else:
            X = load_video(os.path.join(args.video_dir, f"{idx}.mat"), args.frames, args.video_size)
        X = X.cuda()
        eps = args.epsilon if args.epsilon is not None else args.eps_per_entry * X.numel()
        ranks = args.ranks if args.ranks else [args.rank] * X.dim()  # ring ranks R_1..R_N (R_{N+1} = R_1)
        assert len(ranks) == X.dim(), f"--ranks needs {X.dim()} values for shape {tuple(X.shape)}"
        torch.manual_seed(args.seed + idx)  # same cores and E^(0) for every eta
        cores0 = tr.initialize_tr_cores(list(X.shape), ranks + [ranks[0]], device="cuda")
        if args.init == "als10":  # initialize_tr_cores_svd in AT_AdaTR2.py; "random" is AT_TR4video_exp.py
            cores0 = tr.tr_decompose2(X, cores0, max_iter=10, tol=1e-6, dis_num=10 ** 9)
        cores0 = [c.detach() for c in cores0]
        E0 = torch.randn_like(X)
        E0 = E0 / (torch.norm(E0) + 1e-12) * eps ** 0.5
        t0 = time.time()
        out = run_one(X, cores0, E0, eps, args)
        seconds = time.time() - t0
        name = f"{args.kind}{idx}_eta{args.eta:g}"
        np.savez(os.path.join(args.out_dir, name + ".npz"), kind=args.kind, item=idx, shape=np.array(X.shape),
                 eps=eps, eta=args.eta, rank=np.array(ranks), init=args.init, inner_tol=args.inner_tol,
                 update=args.update, seconds=seconds, **out)
        K, rho_bar = out["K"], out["rho"].mean()
        msg = (f"[fag-conv] {name} ({args.update}) shape={tuple(X.shape)} eps={eps:.1f} K={K} rho_bar={rho_bar:.3e} "
               f"recon {out['recon'][0]:.4f}->{out['recon'][-1]:.4f} max_ideal_err={out['ideal_err'].max():.1e} "
               f"min_delta={out['delta'].min():.3e} max_step_err={out['step_err'].max():.1e} "
               f"min_item1={out['item1_slack'].min():.1e} min_C1={out['c1_slack'].min():.1e} time={seconds:.0f}s")
        if args.grad_check:
            msg += (f" t*={out['tstar']} eps_g={out['eps_g']:.3e} |grad g|={out['grad_g_norm']:.3e} "
                    f"|grad g~|={out['grad_gtilde_norm']:.3e} cos={out['cos']:.3f} |G_eta|={out['G_eta_norm']:.3e} "
                    f"cos_dir={out['cos_dir']:.3f} | at K-1: eps_g={out['last_eps_g']:.3e} "
                    f"|grad g|={out['last_grad_g_norm']:.3e} cos={out['last_cos']:.3f} "
                    f"fd/ad={out['last_fd_grad_g'][0] / out['last_fd_grad_g'][1]:.3f}")
        print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize", metavar="FOLDER")
    parser.add_argument("--kind", choices=["image", "video"], default="image")
    parser.add_argument("--items", nargs="+", default=["1"])
    parser.add_argument("--image-dir", default=os.path.join(HERE, "My picture"))
    parser.add_argument("--video-dir", default=os.path.join(HERE, "video_data"))
    parser.add_argument("--resize-width", type=int, default=None, help="proportional image resize (None = full size)")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--video-size", type=int, nargs=2, default=None, metavar=("WIDTH", "HEIGHT"),
                        help="resize video frames (default: native size)")
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--ranks", type=int, nargs="+", default=None, help="per-mode ring ranks, e.g. 5 5 3 3")
    parser.add_argument("--init", choices=["als10", "random"], default="als10")
    parser.add_argument("--update", choices=["jacobi", "sequential"], default="jacobi",
                        help="jacobi = attack_methods.py / Eq. (19); sequential = old main scripts (E overwritten per mode)")
    parser.add_argument("--epsilon", type=float, default=None, help="absolute budget on ||E||_F^2")
    parser.add_argument("--eps-per-entry", type=float, default=0.0005, help="epsilon = this * numel (AT_AdaTR2.py)")
    parser.add_argument("--eta", type=float, default=1.0, help="step that multiplies the gradient (10 * config.lr)")
    parser.add_argument("--outer", type=int, default=100)
    parser.add_argument("--inner", type=int, default=100)
    parser.add_argument("--inner-tol", type=float, default=1e-10)
    parser.add_argument("--outer-tol", type=float, default=1e-6)
    parser.add_argument("--no-grad-check", dest="grad_check", action="store_false")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=os.path.join(HERE, "fag_convergence_results"))
    args = parser.parse_args()
    if args.summarize:
        from fag_convergence_summary import summarize
        summarize(args.summarize)
    else:
        run(args)


if __name__ == "__main__":
    main()
