import argparse
import math
import os
import sys
import torch
import tensorly as tl

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import TR_func as tr


def build_reference_tensor(shape, ranks, device, seed=0):
    torch.manual_seed(seed)
    ranks_extended = list(ranks) + [ranks[0]]
    cores = tr.initialize_tr_cores(shape, ranks_extended, device=device)
    tensor = tl.tr_to_tensor(cores)
    return tensor.to(device), cores


def reconstruct_from_cores(cores):
    with torch.no_grad():
        return tl.tr_to_tensor(cores)


def run_single_test(shape, ranks, max_iter, tol, device, seed=0):
    reference_tensor, initial_cores = build_reference_tensor(shape, ranks, device, seed=seed)
    reference_tensor = reference_tensor.clone().detach()

    cores_init_2 = [core.clone().detach() for core in initial_cores]
    cores_init_4 = [core.clone().detach() for core in initial_cores]

    cores_tr2 = tr.tr_decompose2(
        reference_tensor,
        cores_init_2,
        max_iter=max_iter,
        tol=tol,
        dis_num=max(1, max_iter // 5),
        device=device,
    )
    cores_tr4 = tr.tr_decompose4(
        reference_tensor,
        cores_init_4,
        max_iter=max_iter,
        tol=tol,
        dis_num=max(1, max_iter // 5),
        device=device,
        track_grad=True,
    )

    recon_tr2 = reconstruct_from_cores(cores_tr2)
    recon_tr4 = reconstruct_from_cores(cores_tr4)

    err_tr2 = torch.norm(reference_tensor - recon_tr2) / torch.norm(reference_tensor)
    err_tr4 = torch.norm(reference_tensor - recon_tr4) / torch.norm(reference_tensor)
    diff_tr24 = torch.norm(recon_tr2 - recon_tr4) / torch.norm(reference_tensor)

    stats = {
        "shape": shape,
        "ranks": ranks,
        "error_tr2": err_tr2.item(),
        "error_tr4": err_tr4.item(),
        "diff_tr2_tr4": diff_tr24.item(),
    }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Compare tr_decompose2 and tr_decompose4 reconstructions.")
    parser.add_argument("--shape", type=int, nargs="+", default=[10, 12, 8], help="Tensor shape.")
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 5, 4], help="Tensor ring ranks (cyclic).")
    parser.add_argument("--max-iter", type=int, default=10, help="ALS iterations for each routine.")
    parser.add_argument("--tol", type=float, default=1e-10, help="Convergence tolerance.")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu or cuda).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    args = parser.parse_args()

    tl.set_backend("pytorch")

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    stats = run_single_test(
        tuple(args.shape),
        tuple(args.ranks),
        max_iter=args.max_iter,
        tol=args.tol,
        device=device,
        seed=args.seed,
    )

    print("=== Tensor Ring Decomposition Comparison ===")
    print(f"shape           : {stats['shape']}")
    print(f"ranks           : {stats['ranks']}")
    print(f"device          : {device}")
    print(f"error (tr_decompose2) : {stats['error_tr2']:.6e}")
    print(f"error (tr_decompose4) : {stats['error_tr4']:.6e}")
    print(f"relative diff (TN_G2 vs TN_G4) : {stats['diff_tr2_tr4']:.6e}")


if __name__ == "__main__":
    main()
