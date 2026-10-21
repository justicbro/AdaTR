import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
import torch
from PIL import Image

import TR_func as tr
from attack_angle_experiment import clone_tr_cores, initialize_tr_cores_svd
from attack_methods import (
    objective_function_ATTR,
    optimize_perturbation,
    optimize_perturbation_comparison,
    optimize_perturbation_simplifed,
)

tl.set_backend("pytorch")


@dataclass
class ExperimentConfig:
    epsilon: float
    lr: float
    outer_num_steps_attack: int
    outer_num_steps_defence: int
    inner_num_steps: int
    outer_dis_num: int
    inner_dis_num: int
    inner_tol: float
    attr_epsilon: Optional[float] = None
    unroll_steps: Optional[int] = None
    attack_lr: Optional[float] = None
    defence_lr: Optional[float] = None
    epsilon_per_entry: Optional[float] = None
    theta_epsilon: Optional[float] = None
    outer_tol: float = 1e-6


def load_image_as_tensor(path: str, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image).astype(np.float32) / 255.0
    return torch.tensor(arr, dtype=torch.float32, device=device)


def to_display_image(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3:
        # If channel-first, convert to channel-last.
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
    else:
        raise ValueError(f"Unsupported tensor shape for display: {arr.shape}")
    return np.clip(arr, 0.0, 1.0)


def add_gaussian_noise(X: torch.Tensor, epsilon: float) -> torch.Tensor:
    E = torch.randn_like(X, device=X.device)
    if torch.norm(E) > 0:
        E = E / (torch.norm(E) + 1e-12) * (epsilon ** 0.5)
    return E


def low_rank_nuclear_penalty(E: torch.Tensor, modes: int = 1) -> torch.Tensor:
    """Average nuclear norm over first `modes` unfoldings."""
    if E.dim() == 0:
        return torch.tensor(0.0, device=E.device)
    use_modes = max(1, min(modes, E.dim()))
    penalties: List[torch.Tensor] = []
    for mode in range(use_modes):
        perm = [mode] + [i for i in range(E.dim()) if i != mode]
        E_mode = E.permute(perm).reshape(E.shape[mode], -1)
        s = torch.linalg.svdvals(E_mode)
        penalties.append(torch.sum(s))
    return torch.mean(torch.stack(penalties))


def optimize_perturbation_lr_adatr(
    X: torch.Tensor,
    G_list_ori: List[torch.Tensor],
    config: ExperimentConfig,
    lr_lambda: float = 1e-3,
    lr_modes: int = 1,
    initial_E: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """LR-AdaTR: Ours objective + low-rank penalty on perturbation."""
    if initial_E is not None:
        E = initial_E.clone().detach().to(X.device)
    else:
        E = torch.randn_like(X, device=X.device)
    E.requires_grad_(True)
    optimizer = torch.optim.Adam([E], lr=(config.attack_lr or config.lr))
    loss_curve: List[float] = []

    # Project to epsilon-ball at initialization.
    norm_E = torch.norm(E.detach())
    if norm_E * norm_E > config.epsilon and norm_E > 0:
        E.data.mul_((config.epsilon ** 0.5) / (norm_E + 1e-12))

    TN_G = X.detach().clone()
    for _ in range(config.outer_num_steps_attack):
        optimizer.zero_grad()
        attack_obj, TN_G, _ = objective_function_ATTR(
            X, G_list_ori, E, config.inner_num_steps, config.inner_tol
        )
        lr_penalty = low_rank_nuclear_penalty(E, modes=lr_modes)
        # Minimize: -recon_error^2 + lambda * rank_surrogate
        # -> maximize reconstruction error while encouraging low-rank E.
        total_loss = attack_obj + lr_lambda * lr_penalty
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            norm_E = torch.norm(E)
            if norm_E * norm_E > config.epsilon and norm_E > 0:
                E.data.mul_((config.epsilon ** 0.5) / (norm_E + 1e-12))
            recon_err = torch.norm(X - TN_G).item()
            loss_curve.append(recon_err)

    with torch.no_grad():
        _, TN_final, _ = objective_function_ATTR(
            X, G_list_ori, E, config.inner_num_steps, config.inner_tol
        )
    return E.detach(), TN_final.detach(), loss_curve


def save_metrics_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_compare(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This script currently requires CUDA because attack_methods uses CUDA internally.")
    device = torch.device("cuda")
    os.makedirs(args.output_dir, exist_ok=True)

    X = load_image_as_tensor(args.image_path, device=device)
    config = ExperimentConfig(
        epsilon=args.epsilon,
        lr=args.attack_lr,
        attack_lr=args.attack_lr,
        defence_lr=args.defence_lr,
        outer_num_steps_attack=args.outer_steps_attack,
        outer_num_steps_defence=args.outer_steps_defence,
        inner_num_steps=args.inner_steps,
        outer_dis_num=2000,
        inner_dis_num=2000,
        inner_tol=args.inner_tol,
        attr_epsilon=args.epsilon,
        outer_tol=args.outer_tol,
    )

    tr_cores_init = initialize_tr_cores_svd(X, args.tr_rank, device=str(device))
    results: Dict[str, Dict[str, torch.Tensor]] = {}
    metrics: List[Dict[str, float]] = []

    # 1) Clean baseline
    tr_cores_clean = tr.tr_decompose2(
        X, clone_tr_cores(tr_cores_init), max_iter=config.inner_num_steps, tol=config.inner_tol, device=str(device)
    )
    TN_clean = tl.tr_to_tensor(tr_cores_clean)
    E_clean = torch.zeros_like(X, device=device)
    results["Clean"] = {"E": E_clean, "TN": TN_clean}

    # 2) Gaussian
    E_gaussian = add_gaussian_noise(X, config.epsilon)
    tr_cores_gaussian = tr.tr_decompose2(
        X + E_gaussian,
        clone_tr_cores(tr_cores_init),
        max_iter=config.inner_num_steps,
        tol=config.inner_tol,
        device=str(device),
    )
    TN_gaussian = tl.tr_to_tensor(tr_cores_gaussian)
    results["Gaussian"] = {"E": E_gaussian, "TN": TN_gaussian}

    # 3) Ours_F
    E_ours_f, TN_ours_f, _ = optimize_perturbation_simplifed(
        X, clone_tr_cores(tr_cores_init), config
    )
    results["Ours_F"] = {"E": E_ours_f, "TN": TN_ours_f}

    # 4) Ours
    E_ours, TN_ours, _ = optimize_perturbation(
        X, clone_tr_cores(tr_cores_init), config
    )
    results["Ours"] = {"E": E_ours, "TN": TN_ours}

    # 5) ATTR (comparison branch)
    E_attr, TN_attr, _ = optimize_perturbation_comparison(
        X.detach(), clone_tr_cores(tr_cores_init), config, attr_epsilon=config.epsilon
    )
    results["ATTR"] = {"E": E_attr, "TN": TN_attr}

    # 6) New method: LR-AdaTR (no PCA)
    E_lr, TN_lr, _ = optimize_perturbation_lr_adatr(
        X,
        clone_tr_cores(tr_cores_init),
        config,
        lr_lambda=args.lr_lambda,
        lr_modes=args.lr_modes,
    )
    results["LR-AdaTR"] = {"E": E_lr, "TN": TN_lr}

    # Metrics and visualization
    for method, payload in results.items():
        E = payload["E"]
        TN = payload["TN"]
        recon_err = torch.norm(X - TN).item()
        noise_norm = torch.norm(E).item()
        metrics.append(
            {
                "method": method,
                "reconstruction_error": recon_err,
                "noise_norm_fro": noise_norm,
            }
        )
        print(f"{method:>10s} | recon_error={recon_err:.6f} | ||E||_F={noise_norm:.6f}")

    csv_path = os.path.join(args.output_dir, "single_image_metrics.csv")
    save_metrics_csv(csv_path, metrics)

    ordered_methods = ["Clean", "Gaussian", "Ours_F", "Ours", "ATTR", "LR-AdaTR"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for idx, method in enumerate(ordered_methods):
        r, c = divmod(idx, 3)
        ax = axes[r, c]
        TN = results[method]["TN"]
        recon_err = torch.norm(X - TN).item()
        noise_norm = torch.norm(results[method]["E"]).item()
        ax.imshow(to_display_image(TN))
        ax.set_title(f"{method}\nerr={recon_err:.4f}, ||E||={noise_norm:.4f}")
        ax.axis("off")

    fig.suptitle("Single Image Comparison (No PCA): Clean/Gaussian/Ours/ATTR/LR-AdaTR", fontsize=13)
    fig.tight_layout()
    fig_path = os.path.join(args.output_dir, "single_image_reconstruction_compare.png")
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)

    print(f"Saved figure: {fig_path}")
    print(f"Saved metrics: {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-image attack comparison with new LR-AdaTR.")
    parser.add_argument("--image-path", type=str, required=True, help="Path to one input image.")
    parser.add_argument("--output-dir", type=str, default="lr_adatr_single_image_results")
    parser.add_argument("--tr-rank", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=600.0, help="L2 norm budget squared in this codebase style.")
    parser.add_argument("--attack-lr", type=float, default=0.01)
    parser.add_argument("--defence-lr", type=float, default=0.01)
    parser.add_argument("--outer-steps-attack", type=int, default=30)
    parser.add_argument("--outer-steps-defence", type=int, default=30)
    parser.add_argument("--inner-steps", type=int, default=100)
    parser.add_argument("--inner-tol", type=float, default=1e-10)
    parser.add_argument("--outer-tol", type=float, default=1e-6)
    parser.add_argument("--lr-lambda", type=float, default=1e-3, help="Low-rank regularization weight.")
    parser.add_argument("--lr-modes", type=int, default=1, help="How many unfoldings to average for low-rank penalty.")
    return parser


if __name__ == "__main__":
    run_compare(build_parser().parse_args())
