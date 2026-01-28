import argparse
import csv
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorly as tl
import torch
from PIL import Image
import torch.nn.functional as F
from torch.distributions import Laplace
import matplotlib.pyplot as plt

plt.switch_backend("Agg")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import TR_func as tr


tl.set_backend("pytorch")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clone_tr_cores(cores: List[torch.Tensor]) -> List[torch.Tensor]:
    return [core.clone().detach() for core in cores]


@dataclass
class OptimConfig:
    inner_num_steps: int = 100
    inner_tol: float = 1e-10
    inner_dis_num: int = 200
    outer_num_steps_defence: int = 100
    defence_lr: float = 1e-2
    outer_tol: float = 1e-6


def generate_initial_cores(
    shape: List[int],
    ranks: List[int],
    distribution: str,
    std: float,
    seed: int,
    device: torch.device,
) -> List[torch.Tensor]:
    distribution = distribution.lower()
    if std <= 0:
        raise ValueError("std must be positive.")

    set_global_seed(seed)
    cores: List[torch.Tensor] = []
    for i in range(len(shape)):
        core_shape = (ranks[i], shape[i], ranks[i + 1])
        if distribution == "normal":
            tensor = torch.randn(core_shape, device=device) * std
        elif distribution == "uniform":
            limit = math.sqrt(3.0) * std
            tensor = torch.empty(core_shape, device=device).uniform_(-limit, limit)
        elif distribution == "laplace":
            scale = std / math.sqrt(2.0)
            dist = Laplace(
                torch.tensor(0.0, device=device),
                torch.tensor(scale, device=device),
            )
            tensor = dist.sample(core_shape)
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
        cores.append(tensor)
    return cores


def objective_function_DFTR_noisy(
    X: torch.Tensor,
    G_list: List[torch.Tensor],
    E: torch.Tensor,
    *,
    inner_num_steps: int,
    inner_tol: float,
    inner_dis_num: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    Run the DF-ADdaTR alternating least squares on the attacked tensor (X + E)
    and return the negative reconstruction loss, reconstruction tensor, and cores.
    """
    X_noisy = X + E
    cloned_cores = [core.clone() for core in G_list]
    tr_cores = tr.tr_decompose4(
        X_noisy,
        cloned_cores,
        max_iter=inner_num_steps,
        tol=inner_tol,
        dis_num=inner_dis_num,
        device=device.type,
        track_grad=False,
    )
    TN_G = tl.tr_to_tensor(tr_cores)
    loss = -torch.norm(X_noisy - TN_G) ** 2
    return loss, TN_G, tr_cores


def optimize_G_list_noisy(
    X: torch.Tensor,
    G_list_ori: List[torch.Tensor],
    E: torch.Tensor,
    config: OptimConfig,
    *,
    device: torch.device,
    verbose: bool = False,
) -> Tuple[List[torch.Tensor], List[float]]:
    """
    Gradient-based optimisation of TR cores against the noisy observation X + E.
    Mirrors the behaviour of the in-project optimise_G_list_noisy routine.
    """
    G_list: List[torch.Tensor] = []
    trainable_cores: List[torch.Tensor] = []
    for idx, core in enumerate(G_list_ori):
        cloned = core.clone().detach().to(device)
        if idx != 0:
            cloned.requires_grad_(True)
            trainable_cores.append(cloned)
        G_list.append(cloned)

    if not trainable_cores:
        raise ValueError("No trainable cores were found; check tensor-ring ranks.")

    optimizer = torch.optim.Adam(trainable_cores, lr=config.defence_lr)
    attacked_tensor = X + E
    loss_history: List[float] = []

    for step in range(config.outer_num_steps_defence):
        prev_trainable = [core.detach().clone() for core in trainable_cores]
        optimizer.zero_grad()

        loss, TN_G, _ = objective_function_DFTR_noisy(
            X,
            G_list,
            E,
            inner_num_steps=config.inner_num_steps,
            inner_tol=config.inner_tol,
            inner_dis_num=config.inner_dis_num,
            device=device,
        )

        loss = -loss
        loss.backward()
        optimizer.step()

        loss_value = float(loss.item())
        loss_history.append(loss_value)
        recon_norm_attack = torch.norm(attacked_tensor - TN_G).item()
        recon_norm_clean = torch.norm(X - TN_G).item()

        if verbose:
            print(
                f"[Defence-Noisy] Step {step + 1:03d}/{config.outer_num_steps_defence}: "
                f"loss={loss_value:.6f}, recon_attack={recon_norm_attack:.6f}, "
                f"recon_clean={recon_norm_clean:.6f}"
            )

        rel_change = 0.0
        for prev_core, core in zip(prev_trainable, trainable_cores):
            delta = torch.norm(core.detach() - prev_core)
            base = torch.norm(prev_core)
            rel = delta / (base + 1e-12) if base > 0 else delta
            rel_change = max(rel_change, float(rel))

        if rel_change < config.outer_tol:
            if verbose:
                print(
                    f"[Defence-Noisy] Early stop at step {step + 1} "
                    f"(relative change {rel_change:.3e} < {config.outer_tol})"
                )
            break

    return [core.detach() for core in G_list], loss_history


def optimize_shared_G_list(
    tensors: List[torch.Tensor],
    G_list_ori: List[torch.Tensor],
    config: OptimConfig,
    *,
    device: torch.device,
    verbose: bool = False,
) -> Tuple[List[torch.Tensor], List[float]]:
    """
    Jointly optimise TR cores against multiple tensors by sharing the same G initialisation.
    """
    if not tensors:
        raise ValueError("No tensors provided for shared optimisation.")
    shapes = {tuple(tensor.shape) for tensor in tensors}
    if len(shapes) != 1:
        raise ValueError("All tensors must share the same shape for shared optimisation.")

    tensors = [tensor.to(device) for tensor in tensors]
    zero_noises = [torch.zeros_like(tensor, device=device) for tensor in tensors]

    G_list: List[torch.Tensor] = []
    trainable_cores: List[torch.Tensor] = []
    for idx, core in enumerate(G_list_ori):
        cloned = core.clone().detach().to(device)
        if idx != 0:
            cloned.requires_grad_(True)
            trainable_cores.append(cloned)
        G_list.append(cloned)

    if not trainable_cores:
        raise ValueError("No trainable cores were found; check tensor-ring ranks.")

    optimizer = torch.optim.Adam(trainable_cores, lr=config.defence_lr)
    loss_history: List[float] = []

    for step in range(config.outer_num_steps_defence):
        prev_trainable = [core.detach().clone() for core in trainable_cores]
        optimizer.zero_grad()

        total_loss = 0.0
        recon_norms: List[float] = []

        for tensor, zero_noise in zip(tensors, zero_noises):
            loss, TN_G, _ = objective_function_DFTR_noisy(
                tensor,
                G_list,
                zero_noise,
                inner_num_steps=config.inner_num_steps,
                inner_tol=config.inner_tol,
                inner_dis_num=config.inner_dis_num,
                device=device,
            )
            total_loss = total_loss - loss
            if verbose:
                recon_norms.append(torch.norm(tensor - TN_G).item())

        total_loss.backward()
        optimizer.step()

        loss_value = float(total_loss.item())
        loss_history.append(loss_value)

        if verbose and recon_norms:
            recon_str = ", ".join(f"{val:.6f}" for val in recon_norms)
            print(
                f"[Shared-Init] Step {step + 1:03d}/{config.outer_num_steps_defence}: "
                f"loss={loss_value:.6f}, per-image recon=[{recon_str}]"
            )

        rel_change = 0.0
        for prev_core, core in zip(prev_trainable, trainable_cores):
            delta = torch.norm(core.detach() - prev_core)
            base = torch.norm(prev_core)
            rel = delta / (base + 1e-12) if base > 0 else delta
            rel_change = max(rel_change, float(rel))

        if rel_change < config.outer_tol:
            if verbose:
                print(
                    f"[Shared-Init] Early stop at step {step + 1} "
                    f"(relative change {rel_change:.3e} < {config.outer_tol})"
                )
            break

    return [core.detach() for core in G_list], loss_history


def optimize_missing_initialization(
    X: torch.Tensor,
    mask: torch.Tensor,
    cores_ori: List[torch.Tensor],
    config: OptimConfig,
    *,
    device: torch.device,
    verbose: bool = False,
    value_range: Optional[Tuple[float, float]] = None,
) -> Tuple[torch.Tensor, List[float]]:
    """
    Optimise the initial values assigned to the missing entries of X before ALS.
    """
    X = X.to(device)
    mask = mask.to(device)
    mask_bool = mask.bool()
    if mask_bool.all():
        raise ValueError("Mask does not contain missing entries to optimise.")
    missing_mask = ~mask_bool
    mask_float = mask.to(dtype=X.dtype)

    cores_init = [core.clone().detach().to(device) for core in cores_ori]
    base_recon = tl.tr_to_tensor(cores_init)
    init_missing_values = base_recon.masked_select(missing_mask)
    if init_missing_values.numel() == 0:
        raise ValueError("No missing entries were identified for optimisation.")

    missing_param = torch.nn.Parameter(init_missing_values.clone())
    optimizer = torch.optim.Adam([missing_param], lr=config.defence_lr)

    observed_input = (X * mask_float).detach()
    loss_history: List[float] = []
    grad_norm_history: List[float] = []
    grad_max_history: List[float] = []
    prev_parameter = missing_param.detach().clone()

    for step in range(config.outer_num_steps_defence):
        optimizer.zero_grad()

        full_missing = torch.zeros_like(X)
        full_missing = full_missing.masked_scatter(missing_mask, missing_param)

        completed_tensor, reconstructed, _ = tr.tensor_completion(
            observed_input,
            mask,
            cores_init,
            max_iter=config.inner_num_steps,
            tol=config.inner_tol,
            dis_num=config.inner_dis_num,
            device=device.type,
            missing_init=full_missing,
        )

        masked_error = (reconstructed - X) * mask_float
        loss = torch.norm(masked_error) ** 2
        loss_value = float(loss.item())
        loss.backward()
        if missing_param.grad is not None:
            grad_tensor = missing_param.grad.detach()
            grad_norm_history.append(float(torch.norm(grad_tensor).item()))
            grad_max_history.append(float(torch.max(torch.abs(grad_tensor)).item()))
        else:
            grad_norm_history.append(float("nan"))
            grad_max_history.append(float("nan"))
        optimizer.step()

        if value_range is not None:
            low, high = value_range
            with torch.no_grad():
                missing_param.clamp_(low, high)

        loss_history.append(loss_value)

        with torch.no_grad():
            delta = torch.norm(missing_param.detach() - prev_parameter)
            base = torch.norm(prev_parameter)
            rel_change = float(delta / (base + 1e-12) if base > 0 else delta)
            prev_parameter = missing_param.detach().clone()

        if verbose:
            print(
                f"[Completion-Init] Step {step + 1:03d}/{config.outer_num_steps_defence}: "
                f"loss={loss_value:.6f}, rel_change={rel_change:.3e}"
            )

        if rel_change < config.outer_tol:
            if verbose:
                print(
                    f"[Completion-Init] Early stop at step {step + 1} "
                    f"(relative change {rel_change:.3e} < {config.outer_tol})"
                )
            break

    with torch.no_grad():
        final_missing = torch.zeros_like(X)
        final_missing = final_missing.masked_scatter(missing_mask, missing_param.detach())

    final_missing_combined = observed_input + final_missing

    history = {
        "loss": loss_history,
        "grad_norm": grad_norm_history,
        "grad_max": grad_max_history,
    }

    return final_missing, final_missing_combined, history


def generate_random_mask(
    shape: List[int],
    observed_ratio: float,
    seed: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not 0.0 < observed_ratio < 1.0:
        raise ValueError("observed_ratio must lie in (0, 1) to ensure missing entries exist.")
    set_global_seed(seed)
    random_tensor = torch.rand(shape, device=device)
    mask_bool = random_tensor < observed_ratio
    if mask_bool.all():
        mask_bool.view(-1)[0] = False
    if not mask_bool.any():
        mask_bool.view(-1)[0] = True
    return mask_bool.to(dtype=dtype)


def compute_psnr(reference: torch.Tensor, estimate: torch.Tensor, data_range: float) -> float:
    mse = torch.mean((reference - estimate) ** 2).item()
    if mse <= 0.0:
        return float("inf")
    data_range = max(data_range, 1e-12)
    return 20.0 * math.log10(data_range / math.sqrt(mse))


def _prepare_image_4d(tensor: torch.Tensor) -> torch.Tensor:
    arr = tensor.detach().cpu()
    if arr.ndim == 2:
        arr = arr.unsqueeze(0).unsqueeze(0)
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
            arr = arr.unsqueeze(0)
        else:
            arr = arr.permute(2, 0, 1).unsqueeze(0)
    elif arr.ndim == 4:
        arr = arr
    else:
        raise ValueError(f"Unsupported tensor shape {tuple(arr.shape)} for SSIM computation.")
    return arr.to(dtype=torch.float32)


def _gaussian_kernel(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32)
    center = (window_size - 1) / 2.0
    gauss = torch.exp(-((coords - center) ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel_2d = torch.outer(gauss, gauss)
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel


def compute_ssim(
    reference: torch.Tensor,
    estimate: torch.Tensor,
    data_range: float,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    if data_range <= 0:
        return 1.0

    ref_4d = _prepare_image_4d(reference)
    est_4d = _prepare_image_4d(estimate)
    if ref_4d.shape != est_4d.shape:
        raise ValueError("reference and estimate must share the same shape for SSIM.")

    _, channels, height, width = ref_4d.shape
    if min(height, width) < window_size:
        window_size = min(height, width)
        sigma = max(window_size / 6.0, 1e-3)

    window = _gaussian_kernel(window_size, sigma, channels)
    padding = window_size // 2

    mu1 = F.conv2d(ref_4d, window, padding=padding, groups=channels)
    mu2 = F.conv2d(est_4d, window, padding=padding, groups=channels)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(ref_4d * ref_4d, window, padding=padding, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(est_4d * est_4d, window, padding=padding, groups=channels) - mu2_sq
    sigma12 = F.conv2d(ref_4d * est_4d, window, padding=padding, groups=channels) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(ssim_map.mean().item())


def compute_rse(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    numerator = torch.norm(reference - estimate)
    denominator = torch.norm(reference) + 1e-12
    return float((numerator / denominator).item())


def _format_metric(value: float, width: int = 15, precision: int = 4) -> str:
    if math.isfinite(value):
        return f"{value:>{width}.{precision}f}"
    return f"{str(value):>{width}}"


def print_completion_table(rows: List[Dict[str, Any]], title: str) -> None:
    if title:
        print(title)
    header = f"{'Method':<20}{'PSNR(dB)':>15}{'RSE':>15}{'SSIM':>15}"
    print(header)
    print("-" * len(header))
    for row in rows:
        method = row.get("method", "")
        psnr = row.get("psnr", float("nan"))
        rse = row.get("rse", float("nan"))
        ssim = row.get("ssim", float("nan"))
        print(
            f"{method:<20}"
            f"{_format_metric(psnr)}"
            f"{_format_metric(rse)}"
            f"{_format_metric(ssim)}"
        )
    print()


def save_tensor_as_image(tensor: torch.Tensor, path: str) -> None:
    array = tensor.detach().cpu().clamp(0.0, 1.0).numpy()
    if array.ndim == 2:
        image = (array * 255.0).round().astype(np.uint8)
        Image.fromarray(image).save(path)
    elif array.ndim == 3:
        image = (array * 255.0).round().astype(np.uint8)
        Image.fromarray(image).save(path)
    else:
        raise ValueError(f"Unsupported tensor shape {tuple(array.shape)} for image saving.")


def run_completion_image_experiment(
    X: torch.Tensor,
    mask: torch.Tensor,
    ranks: List[int],
    init_distribution: str,
    init_std: float,
    seed: int,
    config: OptimConfig,
    *,
    device: torch.device,
    verbose: bool = False,
    value_range: Optional[Tuple[float, float]] = None,
    save_root: Optional[str] = None,
    save_images: bool = False,
    save_logs: bool = False,
    image_index: Optional[int] = None,
    image_label: Optional[str] = None,
) -> Dict[str, Any]:
    X = X.to(device)
    mask = mask.to(device)
    mask = mask.to(dtype=X.dtype)

    init_cores = generate_initial_cores(
        list(X.shape),
        ranks,
        distribution=init_distribution,
        std=init_std,
        seed=seed,
        device=device,
    )
    observed_tensor = X * mask
    baseline_completion, baseline_reconstructed, _ = tr.tensor_completion(
        observed_tensor,
        mask,
        init_cores,
        max_iter=config.inner_num_steps,
        tol=config.inner_tol,
        dis_num=config.inner_dis_num,
        device=device.type,
    )

    optim_missing_init, optim_missing_filled, history = optimize_missing_initialization(
        X,
        mask,
        init_cores,
        config,
        device=device,
        verbose=verbose,
        value_range=value_range,
    )
    loss_history = history["loss"]

    optim_completion, optim_reconstructed, _ = tr.tensor_completion(
        observed_tensor,
        mask,
        init_cores,
        max_iter=config.inner_num_steps,
        tol=config.inner_tol,
        dis_num=config.inner_dis_num,
        device=device.type,
        missing_init=optim_missing_init,
    )

    baseline_error = torch.norm(X - baseline_completion).item()
    optim_error = torch.norm(X - optim_completion).item()
    observed_ratio = float(mask.mean().item())
    missing_ratio = 1.0 - observed_ratio

    root_dir = None
    images_dir = None
    logs_dir = None
    image_stub = None
    if save_root and (save_images or save_logs):
        root_dir = os.path.abspath(save_root)
        os.makedirs(root_dir, exist_ok=True)
        if save_images:
            images_dir = os.path.join(root_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
        if save_logs:
            logs_dir = os.path.join(root_dir, "logs")
            os.makedirs(logs_dir, exist_ok=True)
        label_part = ""
        if image_label:
            label_part = os.path.splitext(os.path.basename(image_label))[0].replace(" ", "_")
        idx_part = f"img{image_index}" if image_index is not None else "img"
        miss_part = f"mr{missing_ratio:.3f}".replace(".", "_")
        components = [idx_part]
        if label_part:
            components.append(label_part)
        components.append(miss_part)
        image_stub = "_".join(components)

    if save_logs and logs_dir is not None and image_stub is not None:
        log_path = os.path.join(logs_dir, f"{image_stub}_loss_grad.csv")
        with open(log_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["step", "loss", "grad_norm", "grad_max"])
            for step_idx, (loss_val, grad_norm, grad_max) in enumerate(
                zip(
                    history.get("loss", []),
                    history.get("grad_norm", []),
                    history.get("grad_max", []),
                ),
                start=1,
            ):
                writer.writerow([step_idx, loss_val, grad_norm, grad_max])

    if save_images and images_dir is not None and image_stub is not None:
        save_tensor_as_image(X, os.path.join(images_dir, f"{image_stub}_target.png"))
        save_tensor_as_image(observed_tensor, os.path.join(images_dir, f"{image_stub}_observed.png"))
        save_tensor_as_image(baseline_completion, os.path.join(images_dir, f"{image_stub}_baseline.png"))
        save_tensor_as_image(optim_completion, os.path.join(images_dir, f"{image_stub}_optim.png"))
        save_tensor_as_image(optim_missing_init, os.path.join(images_dir, f"{image_stub}_missing_only.png"))
        save_tensor_as_image(optim_missing_filled, os.path.join(images_dir, f"{image_stub}_missing_filled.png"))

        if loss_history:
            plt.figure(figsize=(6, 4))
            plt.plot(range(1, len(loss_history) + 1), loss_history, marker="o")
            plt.xlabel("Step")
            plt.ylabel("Loss (observed MSE)")
            plt.title("Completion Init Optimisation Loss")
            plt.grid(True, linestyle="--", linewidth=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(images_dir, f"{image_stub}_loss_curve.png"))
            plt.close()

    return {
        "baseline_completion": baseline_completion.detach(),
        "baseline_reconstructed": baseline_reconstructed.detach(),
        "optim_completion": optim_completion.detach(),
        "optim_reconstructed": optim_reconstructed.detach(),
        "baseline_error": baseline_error,
        "optim_error": optim_error,
        "loss_history": loss_history,
        "opt_history": history,
        "optimized_missing_init": optim_missing_init.detach(),
        "optimized_missing_filled": optim_missing_filled.detach(),
        "observed_tensor": observed_tensor.detach(),
        "observed_ratio": observed_ratio,
        "missing_ratio": missing_ratio,
    }


def run_experiment(
    X: torch.Tensor,
    ranks: List[int],
    distributions: List[str],
    init_std_values: List[float],
    base_seed: int,
    config: OptimConfig,
    *,
    device: torch.device,
    verbose: bool = False,
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    zero_noise = torch.zeros_like(X)

    combo_index = 0
    for distribution in distributions:
        for std in init_std_values:
            seed = base_seed + combo_index
            combo_index += 1

            init_cores = generate_initial_cores(
                list(X.shape),
                ranks,
                distribution=distribution,
                std=std,
                seed=seed,
                device=device,
            )

            als_cores = tr.tr_decompose4(
                X,
                clone_tr_cores(init_cores),
                max_iter=config.inner_num_steps,
                tol=config.inner_tol,
                dis_num=config.inner_dis_num,
                device=device.type,
                track_grad=False,
            )
            TN_als = tl.tr_to_tensor(als_cores)
            als_error = torch.norm(X - TN_als).item()

            optimised_cores, loss_history = optimize_G_list_noisy(
                X,
                init_cores,
                zero_noise,
                config,
                device=device,
                verbose=verbose,
            )

            with torch.no_grad():
                _, TN_opt, _ = objective_function_DFTR_noisy(
                    X,
                    optimised_cores,
                    zero_noise,
                    inner_num_steps=config.inner_num_steps,
                    inner_tol=config.inner_tol,
                    inner_dis_num=config.inner_dis_num,
                    device=device,
                )
            opt_error = torch.norm(X - TN_opt).item()

            results.append(
                {
                    "distribution": distribution,
                    "std": std,
                    "als_error": als_error,
                    "opt_error": opt_error,
                    "final_loss": loss_history[-1] if loss_history else float("nan"),
                }
            )

    return results


def run_seed_sweep(
    X: torch.Tensor,
    ranks: List[int],
    base_seed: int,
    count: int,
    distribution: str,
    std: float,
    config: OptimConfig,
    *,
    device: torch.device,
) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []
    distribution = distribution.lower()
    for idx in range(count):
        seed = base_seed + idx
        init_cores = generate_initial_cores(
            list(X.shape),
            ranks,
            distribution=distribution,
            std=std,
            seed=seed,
            device=device,
        )
        als_cores = tr.tr_decompose4(
            X,
            clone_tr_cores(init_cores),
            max_iter=config.inner_num_steps,
            tol=config.inner_tol,
            dis_num=config.inner_dis_num,
            device=device.type,
            track_grad=False,
        )
        TN = tl.tr_to_tensor(als_cores)
        error = torch.norm(X - TN).item()
        results.append(
            {
                "seed": seed,
                "error": error,
                "distribution": distribution,
                "std": std,
            }
        )
    return results


def evaluate_tr_error_with_init(
    tensor: torch.Tensor,
    init_cores: List[torch.Tensor],
    config: OptimConfig,
    *,
    device: torch.device,
) -> float:
    als_cores = tr.tr_decompose4(
        tensor,
        clone_tr_cores(init_cores),
        max_iter=config.inner_num_steps,
        tol=config.inner_tol,
        dis_num=config.inner_dis_num,
        device=device.type,
        track_grad=False,
    )
    TN = tl.tr_to_tensor(als_cores)
    return torch.norm(tensor - TN).item()


def print_results_table(results: List[Dict[str, float]], title: Optional[str] = None) -> None:
    if title:
        print(title)
    if not results:
        print("No results recorded.")
        return

    has_distribution = "distribution" in results[0]
    if has_distribution:
        header = (
            f"{'Dist':<12}"
            f"{'Std':>10}"
            f"{'||X-TN|| (ALS)':>20}"
            f"{'||X-TN|| (Opt)':>20}"
            f"{'Final Loss':>15}"
        )
    else:
        header = (
            f"{'Std':<10}"
            f"{'||X-TN|| (ALS)':>20}"
            f"{'||X-TN|| (Opt)':>20}"
            f"{'Final Loss':>15}"
        )
    print(header)
    print("-" * len(header))
    for entry in results:
        std = entry.get("std")
        als_error = entry["als_error"]
        opt_error = entry["opt_error"]
        final_loss = entry["final_loss"]
        if has_distribution:
            dist = entry.get("distribution", "")
            print(
                f"{dist:<12}"
                f"{std:>10.3f}"
                f"{als_error:>20.6f}"
                f"{opt_error:>20.6f}"
                f"{final_loss:>15.6f}"
            )
        else:
            print(
                f"{std:<10.3f}"
                f"{als_error:>20.6f}"
                f"{opt_error:>20.6f}"
                f"{final_loss:>15.6f}"
            )


def print_seed_sweep_table(results: List[Dict[str, float]], title: Optional[str] = None) -> None:
    if title:
        print(title)
    if not results:
        print("No seed sweep results.")
        return
    header = (
        f"{'Seed':<10}"
        f"{'||X-TN||':>20}"
        f"{'Dist':>12}"
        f"{'Std':>10}"
    )
    print(header)
    print("-" * len(header))
    for entry in results:
        seed = entry["seed"]
        error = entry["error"]
        dist = entry.get("distribution", "")
        std = entry.get("std", float("nan"))
        print(
            f"{seed:<10d}"
            f"{error:>20.6f}"
            f"{dist:>12}"
            f"{std:>10.4f}"
        )
    print()


def run_shared_initialization_experiment(
    train_entries: List[Tuple[int, str]],
    train_tensors: List[torch.Tensor],
    test_entry: Optional[Tuple[int, str]],
    test_tensor: Optional[torch.Tensor],
    ranks: List[int],
    init_distribution: str,
    init_std: float,
    seed: int,
    config: OptimConfig,
    *,
    device: torch.device,
    verbose: bool = False,
) -> Dict[str, Any]:
    if not train_tensors:
        raise ValueError("At least one training tensor is required for the shared initialisation experiment.")
    base_shape = train_tensors[0].shape
    for tensor in train_tensors[1:]:
        if tensor.shape != base_shape:
            raise ValueError("All training tensors must have identical shapes.")
    if test_tensor is not None and test_tensor.shape != base_shape:
        raise ValueError("Test tensor shape must match training tensors for shared initialisation.")

    init_cores = generate_initial_cores(
        list(base_shape),
        ranks,
        distribution=init_distribution,
        std=init_std,
        seed=seed,
        device=device,
    )

    train_baselines: List[float] = []
    for tensor in train_tensors:
        error = evaluate_tr_error_with_init(tensor, init_cores, config, device=device)
        train_baselines.append(error)

    test_baseline: Optional[float] = None
    if test_tensor is not None:
        test_baseline = evaluate_tr_error_with_init(test_tensor, init_cores, config, device=device)

    shared_cores, loss_history = optimize_shared_G_list(
        train_tensors,
        init_cores,
        config,
        device=device,
        verbose=verbose,
    )

    train_shared: List[float] = []
    for tensor in train_tensors:
        error = evaluate_tr_error_with_init(tensor, shared_cores, config, device=device)
        train_shared.append(error)

    test_shared: Optional[float] = None
    if test_tensor is not None:
        test_shared = evaluate_tr_error_with_init(test_tensor, shared_cores, config, device=device)

    train_records: List[Dict[str, Any]] = []
    for (idx, path), baseline, shared in zip(train_entries, train_baselines, train_shared):
        train_records.append(
            {
                "split": "Train",
                "index": idx,
                "path": path,
                "baseline_error": baseline,
                "shared_error": shared,
            }
        )

    test_record: Optional[Dict[str, Any]] = None
    if test_entry is not None and test_baseline is not None and test_shared is not None:
        test_idx, test_path = test_entry
        test_record = {
            "split": "Test",
            "index": test_idx,
            "path": test_path,
            "baseline_error": test_baseline,
            "shared_error": test_shared,
        }

    return {
        "train_records": train_records,
        "test_record": test_record,
        "loss_history": loss_history,
    }


def _format_shared_metric(value: float) -> str:
    return f"{value:>20.6f}"


def print_shared_initialization_table(
    train_records: List[Dict[str, Any]],
    test_record: Optional[Dict[str, Any]],
    title: str,
) -> None:
    print(title)
    header = (
        f"{'Split':<10}"
        f"{'Image':<20}"
        f"{'||X-TN|| Baseline':>20}"
        f"{'||X-TN|| Shared':>20}"
        f"{'Delta':>15}"
    )
    print(header)
    print("-" * len(header))

    def print_record(record: Dict[str, Any]) -> None:
        split = record.get("split", "")
        index = record.get("index", "")
        path = os.path.basename(record.get("path", ""))
        img_label = f"{index}:{path}"
        baseline_error = record.get("baseline_error", float("nan"))
        shared_error = record.get("shared_error", float("nan"))
        delta = baseline_error - shared_error
        print(
            f"{split:<10}"
            f"{img_label:<20}"
            f"{_format_shared_metric(baseline_error)}"
            f"{_format_shared_metric(shared_error)}"
            f"{delta:>15.6f}"
        )

    for rec in train_records:
        print_record(rec)
    if test_record is not None:
        print_record(test_record)
    print()


def build_tensor(
    shape: List[int],
    ranks: List[int],
    *,
    data_seed: int,
    device: torch.device,
) -> torch.Tensor:
    set_global_seed(data_seed)
    cores = tr.initialize_tr_cores(shape, ranks, device=device.type)
    tensor = tl.tr_to_tensor(cores).to(device=device)
    return tensor


def load_image_tensor(
    image_path: str,
    *,
    device: torch.device,
    resize_shape: Optional[Tuple[int, int]] = None,
    resize_width: Optional[int] = None,
) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    if resize_shape is not None:
        image = image.resize(resize_shape, Image.BILINEAR)
    elif resize_width is not None:
        orig_width, orig_height = image.size
        if orig_width == 0:
            raise ValueError(f"Image width is zero for {image_path}")
        scale = resize_width / orig_width
        new_height = max(1, int(round(orig_height * scale)))
        image = image.resize((resize_width, new_height), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    if array.max() > 1.0:
        array = array / 255.0
    tensor = torch.tensor(array, dtype=torch.float32, device=device)
    return tensor


def resize_tensor_to_match(
    tensor: torch.Tensor,
    target_shape: Tuple[int, int, int],
) -> torch.Tensor:
    if list(tensor.shape) == list(target_shape):
        return tensor
    if tensor.ndim != 3 or len(target_shape) != 3:
        raise ValueError("Only 3D HWC tensors are supported for resizing.")
    target_h, target_w, target_c = target_shape
    if tensor.shape[2] != target_c:
        raise ValueError("Channel dimension mismatch when resizing tensors.")
    chw = tensor.permute(2, 0, 1).unsqueeze(0)
    resized = F.interpolate(
        chw,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).permute(1, 2, 0)


def collect_image_paths(
    folder: str,
    start_index: int,
    end_index: int,
) -> List[Tuple[int, str]]:
    if end_index < start_index:
        raise ValueError("--end-index must be greater than or equal to --start-index")
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Input folder does not exist: {folder}")

    exts = [".png", ".jpg", ".jpeg", ".bmp"]
    collected: List[Tuple[int, str]] = []
    for idx in range(start_index, end_index + 1):
        found_path: Optional[str] = None
        for ext in exts:
            candidate = os.path.join(folder, f"{idx}{ext}")
            if os.path.isfile(candidate):
                found_path = candidate
                break
        if found_path is not None:
            collected.append((idx, found_path))
        else:
            print(f"Image not found, skipping index {idx} in {folder}")
    return collected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare optimise_G_list_noisy initialisations with clean data or images."
    )
    parser.add_argument(
        "--data-source",
        choices=["synthetic", "image"],
        default="synthetic",
        help="选择数据来源：合成张量或彩色图像。",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[20, 20, 10],
        help="Shape of the synthetic tensor X.",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        nargs="+",
        default=[4, 4, 4, 4],
        help="Tensor-ring ranks (length must be len(shape) + 1).",
    )
    parser.add_argument(
        "--input-folder",
        default="./AT_TD/My picture",
        help="当 data-source=image 时，图像所在文件夹。",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="当 data-source=image 时，起始索引 (对应 1.png 等文件名)。",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=1,
        help="当 data-source=image 时，结束索引。",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="当 data-source=image 时，直接缩放到指定宽高。",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=None,
        help="当 data-source=image 时，按给定宽度等比例缩放图像。",
    )
    parser.add_argument(
        "--init-seed",
        type=int,
        default=66,
        help="Global随机种子，用于生成初始核心（不同方差仍复用同一随机方向）。",
    )
    parser.add_argument(
        "--init-stds",
        type=float,
        nargs="+",
        default=[0.1, 0.5, 1.0],
        help="初始化核心的目标标准差列表（不同分布使用同一组 std）。",
    )
    parser.add_argument(
        "--init-distributions",
        type=str,
        nargs="+",
        default=["normal", "uniform", "laplace"],
        help="初始核心所使用的分布类型，可选：normal、uniform、laplace 等。",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=2024,
        help="Seed used to generate the clean tensor X.",
    )
    parser.add_argument(
        "--inner-num-steps",
        type=int,
        default=100,
        help="ALS iterations inside DF-ADdaTR.",
    )
    parser.add_argument(
        "--inner-tol",
        type=float,
        default=1e-10,
        help="Tolerance for the inner ALS routine.",
    )
    parser.add_argument(
        "--inner-dis-num",
        type=int,
        default=20,
        help="Verbose interval for inner ALS (matching project defaults).",
    )
    parser.add_argument(
        "--outer-num-steps",
        type=int,
        default=100,
        help="Number of outer optimisation steps.",
    )
    parser.add_argument(
        "--defence-lr",
        type=float,
        default=1e-2,
        help="Learning rate for the optimiser.",
    )
    parser.add_argument(
        "--outer-tol",
        type=float,
        default=1e-6,
        help="Relative change tolerance for early stopping.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (cpu or cuda). Defaults to cuda if available.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-step optimisation diagnostics.",
    )
    parser.add_argument(
        "--enable-completion-exp",
        action="store_true",
        help="启用基于图像的张量补全初始化优化实验。",
    )
    parser.add_argument(
        "--completion-only",
        action="store_true",
        help="仅运行张量补全实验（跳过 optimise_G_list_noisy 的主实验）。",
    )
    parser.add_argument(
        "--completion-observe-ratio",
        type=float,
        default=0.5,
        help="张量补全实验中观测到的像素比例 (0, 1)。",
    )
    parser.add_argument(
        "--completion-mask-seed",
        type=int,
        default=123,
        help="生成张量补全掩码的随机种子。",
    )
    parser.add_argument(
        "--completion-save-prefix",
        type=str,
        default=None,
        help="若提供，则保存目标、观测、基线和优化后的补全图像。",
    )
    parser.add_argument(
        "--enable-shared-init-exp",
        action="store_true",
        help="启用多图像共享初始核心的优化实验。",
    )
    parser.add_argument(
        "--shared-train-count",
        type=int,
        default=0,
        help="共享初始化实验中用于训练的图像数量。",
    )
    parser.add_argument(
        "--shared-test-index",
        type=int,
        default=None,
        help="共享初始化实验中用于测试的图像索引（不在训练集内）。若不提供，则自动选择剩余的第一张图像。",
    )
    parser.add_argument(
        "--shared-init-only",
        action="store_true",
        help="仅运行共享初始化实验，跳过其它单图像或补全实验。",
    )
    parser.add_argument(
        "--completion-output-root",
        type=str,
        default="./completion_outputs_init_opt",
        help="张量补全实验统一输出根目录（内部包含 images 与 logs 子文件夹）。",
    )
    parser.add_argument(
        "--save-completion-images",
        action="store_true",
        help="保存张量补全相关图像及损失曲线到输出目录。",
    )
    parser.add_argument(
        "--save-completion-logs",
        action="store_true",
        help="保存张量补全优化过程的损失与梯度日志到输出目录。",
    )
    parser.add_argument(
        "--enable-seed-scan",
        action="store_true",
        help="启用多随机种子的 TR 分解误差统计实验。",
    )
    parser.add_argument(
        "--seed-scan-count",
        type=int,
        default=5,
        help="种子扫描的样本数量。",
    )
    parser.add_argument(
        "--seed-scan-base",
        type=int,
        default=1000,
        help="种子扫描的起始种子。",
    )
    parser.add_argument(
        "--seed-scan-distribution",
        type=str,
        default="normal",
        help="种子扫描初始化所使用的分布。",
    )
    parser.add_argument(
        "--seed-scan-std",
        type=float,
        default=0.1,
        help="种子扫描初始化的标准差。",
    )
    parser.add_argument(
        "--seed-scan-only",
        action="store_true",
        help="仅运行种子扫描实验。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.seed_scan_only and not args.enable_seed_scan:
        args.enable_seed_scan = True
    if args.seed_scan_only and args.completion_only:
        raise ValueError("--seed-scan-only 不能与 --completion-only 同时使用。")
    if args.seed_scan_only and args.shared_init_only:
        raise ValueError("--seed-scan-only 不能与 --shared-init-only 同时使用。")
    if args.seed_scan_count <= 0 and args.enable_seed_scan:
        raise ValueError("--seed-scan-count 必须为正整数。")

    if args.completion_only:
        if args.data_source != "image":
            raise ValueError("--completion-only 仅支持 data-source=image。")
        args.enable_completion_exp = True

    device_str = (
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    device = torch.device(device_str)

    config = OptimConfig(
        inner_num_steps=args.inner_num_steps,
        inner_tol=args.inner_tol,
        inner_dis_num=args.inner_dis_num,
        outer_num_steps_defence=args.outer_num_steps,
        defence_lr=args.defence_lr,
        outer_tol=args.outer_tol,
    )
    init_distributions = [dist.lower() for dist in args.init_distributions]
    if args.data_source == "synthetic":
        if args.completion_only:
            raise ValueError("--completion-only 仅用于图像数据。")
        shape = args.shape
        ranks = args.ranks
        if len(ranks) != len(shape) + 1:
            raise ValueError("`ranks` length must equal len(shape) + 1 for tensor-ring tensors.")

        X = build_tensor(shape, ranks, data_seed=args.data_seed, device=device)
        if not args.seed_scan_only:
            results = run_experiment(
                X,
                ranks,
                init_distributions,
                args.init_stds,
                args.init_seed,
                config,
                device=device,
                verbose=args.verbose,
            )
            print_results_table(results, title="Synthetic Tensor Results")
        if args.enable_seed_scan:
            sweep_results = run_seed_sweep(
                X,
                ranks,
                args.seed_scan_base,
                args.seed_scan_count,
                args.seed_scan_distribution,
                args.seed_scan_std,
                config,
                device=device,
            )
            print_seed_sweep_table(
                sweep_results,
                title="Seed Sweep (Synthetic)",
            )
    else:
        resize_shape = tuple(args.resize) if args.resize else None
        resize_width = args.resize_width if resize_shape is None else None
        image_entries = collect_image_paths(
            args.input_folder,
            args.start_index,
            args.end_index,
        )
        if not image_entries:
            print("No images found for the specified range; nothing to run.")
            return

        ranks = args.ranks

        if args.enable_shared_init_exp:
            train_count = args.shared_train_count
            if train_count <= 0:
                raise ValueError("--shared-train-count must be positive when enabling the shared initialisation experiment.")
            if len(image_entries) < train_count:
                raise ValueError("Not enough images to satisfy shared initialisation training count.")
            train_entries = image_entries[:train_count]
            if args.shared_test_index is not None:
                test_entry = next((entry for entry in image_entries if entry[0] == args.shared_test_index), None)
                if test_entry is None:
                    raise ValueError(f"Shared test index {args.shared_test_index} not found in the provided image range.")
                if test_entry in train_entries:
                    raise ValueError("Shared test index must not overlap with the training entries.")
            else:
                remaining_entries = [entry for entry in image_entries if entry not in train_entries]
                test_entry = remaining_entries[0] if remaining_entries else None

            train_tensors: List[torch.Tensor] = []
            for train_idx, train_path in train_entries:
                tensor = load_image_tensor(
                    train_path,
                    device=device,
                    resize_shape=resize_shape,
                    resize_width=resize_width,
                )
                train_tensors.append(tensor)

            target_shape = tuple(train_tensors[0].shape)
            train_tensors = [
                resize_tensor_to_match(tensor, target_shape) for tensor in train_tensors
            ]

            dims = list(train_tensors[0].shape)
            if len(ranks) != len(dims) + 1:
                raise ValueError(
                    f"`ranks` length must equal len(image tensor shape) + 1. Received ranks length {len(ranks)} for tensor shape {dims}."
                )

            test_tensor: Optional[torch.Tensor] = None
            if test_entry is not None:
                _, test_path = test_entry
                test_tensor = load_image_tensor(
                    test_path,
                    device=device,
                    resize_shape=resize_shape,
                    resize_width=resize_width,
                )
                test_tensor = resize_tensor_to_match(test_tensor, target_shape)

            shared_results = run_shared_initialization_experiment(
                train_entries,
                train_tensors,
                test_entry,
                test_tensor,
                ranks,
                init_distribution=init_distributions[0],
                init_std=args.init_stds[0],
                seed=args.init_seed,
                config=config,
                device=device,
                verbose=args.verbose,
            )
            shared_title = (
                f"Shared Init Experiment | Train Count {len(train_entries)}"
                + (f", Test Index {test_entry[0]}" if test_entry else ", No Test Image")
            )
            print_shared_initialization_table(
                shared_results["train_records"],
                shared_results["test_record"],
                title=shared_title,
            )
            if args.shared_init_only:
                return

        for idx, image_path in image_entries:
            image_tensor = load_image_tensor(
                image_path,
                device=device,
                resize_shape=resize_shape,
                resize_width=resize_width,
            )
            dims = list(image_tensor.shape)
            ranks = args.ranks
            if len(ranks) != len(dims) + 1:
                raise ValueError(
                    f"`ranks` length must equal len(image tensor shape) + 1. Received ranks length {len(ranks)} for tensor shape {dims}."
                )

            if not (args.completion_only or args.seed_scan_only):
                results = run_experiment(
                    image_tensor,
                    ranks,
                    init_distributions,
                    args.init_stds,
                    args.init_seed,
                    config,
                    device=device,
                    verbose=args.verbose,
                )
                title = f"Image {idx} ({os.path.basename(image_path)})"
                print_results_table(results, title=title)
            if args.enable_completion_exp and not args.seed_scan_only:
                observed_ratio = args.completion_observe_ratio
                mask = generate_random_mask(
                    dims,
                    observed_ratio,
                    seed=args.completion_mask_seed + idx,
                    device=device,
                    dtype=image_tensor.dtype,
                )
                value_min = float(image_tensor.min().item())
                value_max = float(image_tensor.max().item())
                save_root = (
                    args.completion_output_root
                    if (args.save_completion_images or args.save_completion_logs)
                    else None
                )
                completion_result = run_completion_image_experiment(
                    image_tensor,
                    mask,
                    ranks,
                    init_distribution=init_distributions[0],
                    init_std=args.init_stds[0],
                    seed=args.init_seed + idx,
                    config=config,
                    device=device,
                    verbose=args.verbose,
                    value_range=(value_min, value_max),
                    save_root=save_root,
                    save_images=args.save_completion_images,
                    save_logs=args.save_completion_logs,
                    image_index=idx,
                    image_label=os.path.basename(image_path),
                )
                observed_ratio_actual = completion_result["observed_ratio"]
                missing_rate = completion_result["missing_ratio"]
                data_range = max(value_max - value_min, 1e-12)

                rows = []
                observed_metrics = {
                    "method": "Observed (Masked)",
                    "psnr": compute_psnr(image_tensor, completion_result["observed_tensor"], data_range),
                    "rse": compute_rse(image_tensor, completion_result["observed_tensor"]),
                    "ssim": compute_ssim(image_tensor, completion_result["observed_tensor"], data_range),
                }
                rows.append(observed_metrics)

                baseline_metrics = {
                    "method": "ALS Baseline",
                    "psnr": compute_psnr(image_tensor, completion_result["baseline_completion"], data_range),
                    "rse": compute_rse(image_tensor, completion_result["baseline_completion"]),
                    "ssim": compute_ssim(image_tensor, completion_result["baseline_completion"], data_range),
                }
                rows.append(baseline_metrics)

                optim_metrics = {
                    "method": "Init Optimised",
                    "psnr": compute_psnr(image_tensor, completion_result["optim_completion"], data_range),
                    "rse": compute_rse(image_tensor, completion_result["optim_completion"]),
                    "ssim": compute_ssim(image_tensor, completion_result["optim_completion"], data_range),
                }
                rows.append(optim_metrics)

                title = (
                    f"Image {idx} ({os.path.basename(image_path)}) | "
                    f"Missing Rate {missing_rate:.3f}"
                )
                print_completion_table(rows, title)

                if args.completion_save_prefix:
                    prefix = f"{args.completion_save_prefix}_img{idx}"
                    prefix_dir = os.path.dirname(prefix)
                    if prefix_dir:
                        os.makedirs(prefix_dir, exist_ok=True)
                    save_tensor_as_image(image_tensor, f"{prefix}_target.png")
                    save_tensor_as_image(
                        completion_result["observed_tensor"],
                        f"{prefix}_observed.png",
                    )
                    save_tensor_as_image(
                        completion_result["baseline_completion"],
                        f"{prefix}_baseline.png",
                    )
                    save_tensor_as_image(
                        completion_result["optim_completion"],
                        f"{prefix}_optim.png",
                    )
            if args.enable_seed_scan:
                base_seed = args.seed_scan_base + idx * args.seed_scan_count
                sweep_results = run_seed_sweep(
                    image_tensor,
                    ranks,
                    base_seed,
                    args.seed_scan_count,
                    args.seed_scan_distribution,
                    args.seed_scan_std,
                    config,
                    device=device,
                )
                title = f"Seed Sweep - Image {idx} ({os.path.basename(image_path)})"
                print_seed_sweep_table(sweep_results, title=title)
            print()


if __name__ == "__main__":
    main()
