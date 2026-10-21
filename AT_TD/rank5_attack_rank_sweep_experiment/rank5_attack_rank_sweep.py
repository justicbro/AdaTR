import argparse
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
import torch
from PIL import Image
from skimage.metrics import structural_similarity

# Ensure imports work when running from this subfolder.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
AT_TD_DIR = os.path.dirname(CURRENT_DIR)
if AT_TD_DIR not in sys.path:
    sys.path.insert(0, AT_TD_DIR)

import TR_func as tr
from attack_methods import (
    optimize_perturbation,
    optimize_perturbation_comparison,
    optimize_perturbation_simplifed,
)


tl.set_backend("pytorch")


@dataclass
class OptimizationConfig:
    epsilon: float
    lr: float
    outer_num_steps_attack: int
    outer_num_steps_defence: int
    inner_num_steps: int
    inner_dis_num: int
    inner_tol: float
    attack_lr: float
    defence_lr: float
    attr_epsilon: float
    outer_tol: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_cores(cores: List[torch.Tensor]) -> List[torch.Tensor]:
    return [core.clone().detach() for core in cores]


def init_tr_cores_from_tensor(
    tensor: torch.Tensor,
    rank: int,
    max_iter: int,
    tol: float,
    device: str,
) -> List[torch.Tensor]:
    order = tensor.dim()
    ranks = [rank] * order + [rank]
    init_cores = tr.initialize_tr_cores(list(tensor.shape), ranks, device=device)
    cores = tr.tr_decompose2(
        tensor.detach(),
        init_cores,
        max_iter=max_iter,
        tol=tol,
        dis_num=2000,
        device=device,
    )
    return [c.detach() for c in cores]


def tr_reconstruct_with_rank(
    tensor: torch.Tensor,
    rank: int,
    inner_num_steps: int,
    inner_tol: float,
    device: str,
) -> torch.Tensor:
    cores = init_tr_cores_from_tensor(
        tensor=tensor,
        rank=rank,
        max_iter=inner_num_steps,
        tol=inner_tol,
        device=device,
    )
    return tl.tr_to_tensor(cores).detach()


def load_image_tensor(image_path: str, resize_hw: Tuple[int, int], device: str) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    if resize_hw is not None:
        h, w = resize_hw
        image = image.resize((w, h), Image.BILINEAR)
    image_np = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(image_np).to(device=device, dtype=torch.float32)


def clip01(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)


def make_noise_with_norm(x: torch.Tensor, target_l2: float) -> torch.Tensor:
    noise = torch.randn_like(x)
    norm = torch.norm(noise) + 1e-12
    return noise * (target_l2 / norm)


def psnr_np(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-15:
        return float("inf")
    return 20.0 * np.log10(1.0 / np.sqrt(mse))


def ssim_np(a: np.ndarray, b: np.ndarray) -> float:
    try:
        return float(structural_similarity(a, b, channel_axis=2, data_range=1.0))
    except TypeError:
        return float(structural_similarity(a, b, multichannel=True, data_range=1.0))


def rse_np(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a.reshape(-1))) + 1e-12
    return float(np.linalg.norm((a - b).reshape(-1)) / denom)


def tensor_to_np(x: torch.Tensor) -> np.ndarray:
    # Do not clamp here; compute metrics using the raw values.
    return x.detach().cpu().numpy().astype(np.float32)


def evaluate_metrics(ref: torch.Tensor, est: torch.Tensor) -> Dict[str, float]:
    ref_np = tensor_to_np(ref)
    est_np = tensor_to_np(est)
    return {
        "PSNR": psnr_np(ref_np, est_np),
        "SSIM": ssim_np(ref_np, est_np),
        "RSE": rse_np(ref_np, est_np),
    }


def build_attacked_inputs(
    x: torch.Tensor,
    rank5_cores: List[torch.Tensor],
    cfg: OptimizationConfig,
    device: str,
    methods: List[str],
) -> Dict[str, torch.Tensor]:
    attacked: Dict[str, torch.Tensor] = {}
    method_set = set(methods)

    if "Clean" in method_set:
        attacked["Clean"] = x.detach().clone()

    target_l2 = np.sqrt(cfg.epsilon)
    if "Gaussian" in method_set:
        e_gauss = make_noise_with_norm(x, target_l2)
        attacked["Gaussian"] = (x + e_gauss).detach()

    shared_init = make_noise_with_norm(x, target_l2)

    if "AdaTR" in method_set:
        e_adatr, _, _ = optimize_perturbation(
            x,
            clone_cores(rank5_cores),
            cfg,
            initial_E=shared_init.clone().detach(),
        )
        attacked["AdaTR"] = (x + e_adatr).detach()

    if "FAG-AdaTR" in method_set:
        e_fag, _, _ = optimize_perturbation_simplifed(
            x,
            clone_cores(rank5_cores),
            cfg,
            initial_E=shared_init.clone().detach(),
        )
        attacked["FAG-AdaTR"] = (x + e_fag).detach()

    if "ATTR" in method_set:
        attr_cfg = OptimizationConfig(**vars(cfg))
        attr_cfg.attr_epsilon = cfg.epsilon
        attr_cfg.outer_num_steps_defence = cfg.outer_num_steps_attack
        attr_cfg.defence_lr = cfg.attack_lr
        e_attr, _, _ = optimize_perturbation_comparison(
            x.detach(),
            clone_cores(rank5_cores),
            attr_cfg,
            attr_epsilon=attr_cfg.attr_epsilon,
            clean_tensor=None,
        )
        attacked["ATTR"] = (x + e_attr.to(device)).detach()
    return attacked


def run_rank_sweep(
    x: torch.Tensor,
    attacked_inputs: Dict[str, torch.Tensor],
    rank_list: List[int],
    cfg: OptimizationConfig,
    device: str,
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    results: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for method_name, a_adv in attacked_inputs.items():
        results[method_name] = {
            "vs_X": {"PSNR": [], "SSIM": [], "RSE": []},
            "vs_A_adv": {"PSNR": [], "SSIM": [], "RSE": []},
        }
        print(f"\n[Method] {method_name}")
        for rank in rank_list:
            xhat_adv = tr_reconstruct_with_rank(
                tensor=a_adv,
                rank=rank,
                inner_num_steps=cfg.inner_num_steps,
                inner_tol=cfg.inner_tol,
                device=device,
            )
            m1 = evaluate_metrics(x, xhat_adv)
            m2 = evaluate_metrics(a_adv, xhat_adv)
            for k in ("PSNR", "SSIM", "RSE"):
                results[method_name]["vs_X"][k].append(m1[k])
                results[method_name]["vs_A_adv"][k].append(m2[k])
            print(
                f"  rank={rank:2d} | "
                f"vs_X: PSNR={m1['PSNR']:.3f}, SSIM={m1['SSIM']:.4f}, RSE={m1['RSE']:.5f} | "
                f"vs_A_adv: PSNR={m2['PSNR']:.3f}, SSIM={m2['SSIM']:.4f}, RSE={m2['RSE']:.5f}"
            )
    return results


def image_to_matrix(x: torch.Tensor) -> np.ndarray:
    # Flatten RGB image as H x (W*C) matrix for singular value analysis.
    x_np = tensor_to_np(x)
    h, w, c = x_np.shape
    return x_np.reshape(h, w * c)


def singular_values_from_tensor(x: torch.Tensor) -> np.ndarray:
    mat = image_to_matrix(x)
    return np.linalg.svd(mat, compute_uv=False)


def plot_singular_value_analysis(
    attacked_inputs: Dict[str, torch.Tensor],
    cfg: OptimizationConfig,
    device: str,
    save_path: str,
) -> None:
    methods = ["Clean", "Gaussian", "FAG-AdaTR", "ATTR", "AdaTR"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for method in methods:
        if method not in attacked_inputs:
            continue
        x_adv = attacked_inputs[method]
        xadv_rec = tr_reconstruct_with_rank(
            tensor=x_adv,
            rank=5,
            inner_num_steps=cfg.inner_num_steps,
            inner_tol=cfg.inner_tol,
            device=device,
        )

        sv_adv = singular_values_from_tensor(x_adv)
        sv_rec = singular_values_from_tensor(xadv_rec)
        k_adv = np.arange(1, len(sv_adv) + 1)
        k_rec = np.arange(1, len(sv_rec) + 1)

        axes[0].plot(k_adv, sv_adv, linewidth=1.8, label=method)
        axes[1].plot(k_rec, sv_rec, linewidth=1.8, label=method)

    axes[0].set_title("Singular Values of X_adv")
    axes[1].set_title("Singular Values of Xadv_rec (TR rank=5)")
    for ax in axes:
        ax.set_xlabel("Index")
        ax.set_ylabel("Singular Value")
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def optimal_hard_threshold_coeff(beta: float) -> float:
    """Gavish & Donoho 2014: polynomial approximation of omega(beta)
    for unknown noise level.  threshold = omega * median(singular_values)."""
    return 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43


def spectral_purify_gd(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, float]:
    """Apply Gavish-Donoho optimal hard thresholding on mode-1 unfolding."""
    x_np = tensor_to_np(x)
    h, w, c = x_np.shape
    mat = x_np.reshape(h, w * c)
    m, n = mat.shape
    U, sigma, Vt = np.linalg.svd(mat, full_matrices=False)

    beta = min(m, n) / max(m, n)
    omega = optimal_hard_threshold_coeff(beta)
    threshold = omega * np.median(sigma)

    sigma_filtered = np.where(sigma > threshold, sigma, 0.0)
    mat_filtered = (U * sigma_filtered[np.newaxis, :]) @ Vt
    x_purified_np = mat_filtered.reshape(h, w, c).astype(np.float32)

    return (
        torch.from_numpy(x_purified_np).to(device=x.device, dtype=x.dtype),
        sigma,
        sigma_filtered,
        threshold,
    )


def build_purified_inputs(
    attacked_inputs: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Tuple[np.ndarray, np.ndarray, float]]]:
    """Apply Gavish-Donoho hard thresholding to every attacked input."""
    purified: Dict[str, torch.Tensor] = {}
    sv_info: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
    for method, x_adv in attacked_inputs.items():
        if method == "Clean":
            purified[method] = x_adv.clone()
            sv = singular_values_from_tensor(x_adv)
            sv_info[method] = (sv, sv, 0.0)
        else:
            x_pur, sv_orig, sv_filt, thresh = spectral_purify_gd(x_adv)
            purified[method] = x_pur
            sv_info[method] = (sv_orig, sv_filt, thresh)
            kept = int(np.sum(sv_filt > 0))
            print(
                f"[GD] {method}: threshold={thresh:.4f}, "
                f"kept {kept}/{len(sv_orig)} singular values"
            )
    return purified, sv_info


def plot_sv_before_after(
    sv_info: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    save_path: str,
) -> None:
    """Singular-value spectrum before vs after GD hard thresholding."""
    methods_order = ["Clean", "Gaussian", "FAG-AdaTR", "ATTR", "AdaTR"]
    available = [m for m in methods_order if m in sv_info]
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    for idx, method in enumerate(available):
        ax = axes[0, idx]
        sv_orig, sv_filt, threshold = sv_info[method]
        k = np.arange(1, len(sv_orig) + 1)
        ax.semilogy(k, sv_orig, "b-", linewidth=1.5, label="Before")
        if method != "Clean":
            sv_plot = sv_filt.copy()
            sv_plot[sv_plot <= 0] = np.nan
            ax.semilogy(k, sv_plot, "r-", linewidth=1.5, label="After GD")
            ax.axhline(
                y=threshold, color="g", linestyle="--", alpha=0.7,
                label=f"\u03c4={threshold:.2f}",
            )
        ax.set_title(method)
        ax.set_xlabel("Index")
        ax.set_ylabel("Singular Value")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
    fig.suptitle("Gavish-Donoho Hard Thresholding: Before vs After", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_defense_comparison(
    rank_list: List[int],
    results_original: Dict[str, Dict[str, Dict[str, List[float]]]],
    results_purified: Dict[str, Dict[str, Dict[str, List[float]]]],
    save_path: str,
) -> None:
    """Overlay original (solid) and GD-purified (dashed) metric curves."""
    methods = ["Clean", "Gaussian", "FAG-AdaTR", "ATTR", "AdaTR"]
    colors = {
        "Clean": "tab:blue", "Gaussian": "tab:orange",
        "FAG-AdaTR": "tab:green", "ATTR": "tab:red", "AdaTR": "tab:purple",
    }
    metric_names = ["PSNR", "SSIM", "RSE"]
    groups = [("vs_X", "Xhat vs X"), ("vs_A_adv", "Xhat vs Input")]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for col, metric in enumerate(metric_names):
        for row, (grp_key, grp_title) in enumerate(groups):
            ax = axes[row, col]
            for method in methods:
                c = colors.get(method, "black")
                if method in results_original:
                    ax.plot(
                        rank_list, results_original[method][grp_key][metric],
                        marker="o", linewidth=1.8, color=c, label=method,
                    )
                if method in results_purified and method != "Clean":
                    ax.plot(
                        rank_list, results_purified[method][grp_key][metric],
                        marker="s", linewidth=1.8, color=c, linestyle="--",
                        label=f"{method}+GD",
                    )
            ax.set_title(f"{metric}: {grp_title}")
            ax.set_xlabel("TR Rank")
            ax.set_ylabel(metric)
            ax.grid(True, linestyle="--", alpha=0.4)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=5,
        bbox_to_anchor=(0.5, 1.03), fontsize=8,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_2x3(
    rank_list: List[int],
    results: Dict[str, Dict[str, Dict[str, List[float]]]],
    save_path: str,
) -> None:
    methods = ["Clean", "Gaussian", "FAG-AdaTR", "ATTR", "AdaTR"]
    metric_names = ["PSNR", "SSIM", "RSE"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for col, metric in enumerate(metric_names):
        ax_top = axes[0, col]
        ax_bottom = axes[1, col]
        for method in methods:
            if method not in results:
                continue
            ax_top.plot(rank_list, results[method]["vs_X"][metric], marker="o", linewidth=1.8, label=method)
            ax_bottom.plot(
                rank_list,
                results[method]["vs_A_adv"][metric],
                marker="o",
                linewidth=1.8,
                label=method,
            )
        ax_top.set_title(f"{metric}: Xhat_adv vs X")
        ax_bottom.set_title(f"{metric}: Xhat_adv vs A_adv")
        ax_top.set_xlabel("TR Rank")
        ax_bottom.set_xlabel("TR Rank")
        ax_top.set_ylabel(metric)
        ax_bottom.set_ylabel(metric)
        ax_top.grid(True, linestyle="--", alpha=0.4)
        ax_bottom.grid(True, linestyle="--", alpha=0.4)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank=5 attack, then rank sweep (2..16) TR reconstruction metrics."
    )
    parser.add_argument("--image-path", type=str, default="AT_TD/My picture/1.png")
    parser.add_argument("--output-dir", type=str, default="AT_TD/rank5_attack_rank_sweep_experiment/results")
    parser.add_argument("--output-name", type=str, default="rank5_attack_rank_sweep_2x3.png")
    parser.add_argument("--height", type=int, default=200)
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=40.0)
    parser.add_argument("--attack-lr", type=float, default=0.001)
    parser.add_argument("--attack-steps", type=int, default=100)
    parser.add_argument("--inner-steps", type=int, default=200)
    parser.add_argument("--inner-tol", type=float, default=1e-8)
    parser.add_argument("--outer-tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--gpu-id", type=int, default=0, help="CUDA device index, e.g. 0/1/2.")
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["Clean", "Gaussian", "FAG-AdaTR", "ATTR", "AdaTR"],
        help="Subset of methods to run: Clean Gaussian FAG-AdaTR ATTR AdaTR",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This experiment currently requires CUDA because existing attack methods use CUDA.")
    torch.cuda.set_device(args.gpu_id)
    device = f"cuda:{args.gpu_id}"
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    out_png = os.path.join(args.output_dir, args.output_name)
    out_svd_png = os.path.join(
        args.output_dir,
        f"{os.path.splitext(args.output_name)[0]}_singular_values.png",
    )

    x = load_image_tensor(args.image_path, resize_hw=(args.height, args.width), device=device)
    cfg = OptimizationConfig(
        epsilon=float(args.epsilon),
        lr=float(args.attack_lr),
        outer_num_steps_attack=int(args.attack_steps),
        outer_num_steps_defence=int(args.attack_steps),
        inner_num_steps=int(args.inner_steps),
        inner_dis_num=2000,
        inner_tol=float(args.inner_tol),
        attack_lr=float(args.attack_lr),
        defence_lr=float(args.attack_lr),
        attr_epsilon=float(args.epsilon),
        outer_tol=float(args.outer_tol),
    )

    rank5_cores = init_tr_cores_from_tensor(
        tensor=x,
        rank=5,
        max_iter=cfg.inner_num_steps,
        tol=cfg.inner_tol,
        device=device,
    )

    attacked_inputs = build_attacked_inputs(
        x=x,
        rank5_cores=rank5_cores,
        cfg=cfg,
        device=device,
        methods=args.methods,
    )
    rank_list = list(range(2, 17))
    results = run_rank_sweep(
        x=x,
        attacked_inputs=attacked_inputs,
        rank_list=rank_list,
        cfg=cfg,
        device=device,
    )
    plot_2x3(rank_list=rank_list, results=results, save_path=out_png)
    plot_singular_value_analysis(
        attacked_inputs=attacked_inputs,
        cfg=cfg,
        device=device,
        save_path=out_svd_png,
    )
    print(f"\nSaved combined figure to: {out_png}")
    print(f"Saved singular-value figure to: {out_svd_png}")

    # ---- Gavish-Donoho spectral defense ----
    stem = os.path.splitext(args.output_name)[0]

    print("\n===== Gavish-Donoho Defense =====")
    purified_inputs, sv_info = build_purified_inputs(attacked_inputs)

    print("\n--- Rank sweep on GD-purified inputs ---")
    results_purified = run_rank_sweep(
        x=x,
        attacked_inputs=purified_inputs,
        rank_list=rank_list,
        cfg=cfg,
        device=device,
    )

    out_defense_png = os.path.join(args.output_dir, f"{stem}_defense_gd.png")
    plot_2x3(rank_list=rank_list, results=results_purified, save_path=out_defense_png)
    print(f"Saved GD-defense metrics figure to: {out_defense_png}")

    out_compare_png = os.path.join(args.output_dir, f"{stem}_defense_comparison.png")
    plot_defense_comparison(
        rank_list, results, results_purified, save_path=out_compare_png,
    )
    print(f"Saved defense comparison figure to: {out_compare_png}")

    out_sv_ba_png = os.path.join(args.output_dir, f"{stem}_sv_before_after_gd.png")
    plot_sv_before_after(sv_info, save_path=out_sv_ba_png)
    print(f"Saved SV before/after figure to: {out_sv_ba_png}")


if __name__ == "__main__":
    main()
