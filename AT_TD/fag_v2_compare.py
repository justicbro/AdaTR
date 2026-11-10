"""Compare FAG-AdaTR v1 (G^(T) only) with v2 (paper gradient with G^(T-1) and G^(T)) on image TR decomposition.

Setup follows epsilon_comp3.py: images resized to 128x128x3, random TR cores, ALS from the same
initial cores for every method, final error ||X - TR(ALS(X + E))||_F.

Example (from the project root):
    python AT_TD/fag_v2_compare.py --rank 5 --images 1 2 --epsilons 5 25 45 66 --out AT_TD/fag_v2_compare_results/rank5_a.csv
"""
import argparse
import csv
import os
import time
from dataclasses import dataclass

import numpy as np
import tensorly as tl
import torch
from PIL import Image

import TR_func as tr
from attack_methods import optimize_perturbation_simplifed, optimize_perturbation_simplifed_v2

tl.set_backend("pytorch")


@dataclass
class Config:
    epsilon: float
    lr: float
    outer_num_steps_attack: int
    inner_num_steps: int
    inner_tol: float
    inner_dis_num: int = 2000
    outer_tol: float = 1e-6


def load_image(path, device):
    image = Image.open(path).convert("RGB").resize((128, 128), Image.LANCZOS)
    return torch.tensor(np.array(image) / 255.0, dtype=torch.float32, device=device)


def final_error(X, E, cores_init, cfg):
    cores = tr.tr_decompose2(X + E, cores_init, max_iter=cfg.inner_num_steps, tol=cfg.inner_tol, dis_num=10**9)
    return torch.norm(X - tl.tr_to_tensor(cores)).item()


METHODS = {
    "v1": lambda X, G, cfg, E0: optimize_perturbation_simplifed(X, G, cfg, initial_E=E0),
    "v2_sweep": lambda X, G, cfg, E0: optimize_perturbation_simplifed_v2(X, G, cfg, initial_E=E0, prev_mode="sweep"),
    "v2_als": lambda X, G, cfg, E0: optimize_perturbation_simplifed_v2(X, G, cfg, initial_E=E0, prev_mode="als"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="./AT_TD/My picture")
    parser.add_argument("--images", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[5, 25, 45, 66])
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--lr", type=float, default=0.05, help="base lr; both FAG versions multiply it by 10")
    parser.add_argument("--outer-steps", type=int, default=100)
    parser.add_argument("--inner-steps", type=int, default=100)
    parser.add_argument("--inner-tol", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    device = "cuda"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows, curves = [], {}
    for idx in args.images:
        X = load_image(os.path.join(args.image_dir, f"{idx}.png"), device)
        torch.manual_seed(args.seed + idx)
        cores_init = tr.initialize_tr_cores(list(X.shape), [args.rank] * (X.dim() + 1), device=device)
        E0 = torch.randn_like(X)
        base = dict(image=idx, rank=args.rank)
        cfg = Config(epsilon=0.0, lr=args.lr, outer_num_steps_attack=args.outer_steps,
                     inner_num_steps=args.inner_steps, inner_tol=args.inner_tol)
        rows.append(dict(base, epsilon=0, method="clean", error=final_error(X, torch.zeros_like(X), cores_init, cfg),
                         steps=0, seconds=0.0, E_norm=0.0))
        for eps in args.epsilons:
            cfg.epsilon = eps
            E_gauss = E0 / torch.norm(E0) * eps ** 0.5
            rows.append(dict(base, epsilon=eps, method="gauss", error=final_error(X, E_gauss, cores_init, cfg),
                             steps=0, seconds=0.0, E_norm=torch.norm(E_gauss).item()))
            for name in args.methods:
                torch.cuda.synchronize()
                t0 = time.time()
                E, _, loss_list = METHODS[name](X, cores_init, cfg, E0)
                torch.cuda.synchronize()
                seconds = time.time() - t0
                err = final_error(X, E, cores_init, cfg)
                rows.append(dict(base, epsilon=eps, method=name, error=err, steps=len(loss_list),
                                 seconds=seconds, E_norm=torch.norm(E).item()))
                curves[f"img{idx}_eps{eps:g}_{name}"] = np.asarray(loss_list)
                print(f"[compare] img={idx} rank={args.rank} eps={eps:g} {name}: error={err:.4f} "
                      f"steps={len(loss_list)} time={seconds:.1f}s", flush=True)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    np.savez(os.path.splitext(args.out)[0] + "_curves.npz", **curves)


if __name__ == "__main__":
    main()
