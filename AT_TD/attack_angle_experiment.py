import argparse
import csv
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorly as tl
import torch
from PIL import Image

import TR_func as tr
from attack_methods import (
    compute_linf_bound,
    optimize_perturbation,
    optimize_perturbation_comparison,
    optimize_perturbation_simplifed,
)

tl.set_backend("pytorch")


@dataclass
class EvalConfig:
    epsilon: float
    attack_lr: float
    defence_lr: float
    outer_num_steps_attack: int
    outer_num_steps_defence: int
    inner_num_steps: int
    inner_dis_num: int
    inner_tol: float
    outer_tol: float
    attr_epsilon: Optional[float] = None
    epsilon_per_entry: Optional[float] = None


def clone_tr_cores(cores: List[torch.Tensor]) -> List[torch.Tensor]:
    return [core.clone().detach() for core in cores]


def initialize_tr_cores_svd(X: torch.Tensor, rank_value: int, device: str = "cuda") -> List[torch.Tensor]:
    shape = list(X.shape)
    ranks = [rank_value for _ in shape] + [rank_value]
    cores = tr.initialize_tr_cores(shape, ranks, device=device)
    X_detached = X.detach()
    cores = tr.tr_decompose2(
        X_detached,
        cores,
        max_iter=50,
        tol=1e-6,
        device=device,
    )
    return [core.detach() for core in cores]


def extract_tr_mode_rank_products(tr_cores: List[torch.Tensor]) -> List[int]:
    products: List[int] = []
    for core in tr_cores:
        if core.dim() != 3:
            products.append(0)
            continue
        r_i = core.shape[0]
        r_ip1 = core.shape[2]
        products.append(int(r_i * r_ip1))
    return products


def _mode_unfold(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    if tensor.dim() == 0:
        return tensor.reshape(1, 1)
    mode = mode % tensor.dim()
    perm_order = [mode] + [i for i in range(tensor.dim()) if i != mode]
    permuted = tensor.permute(perm_order)
    return permuted.reshape(tensor.shape[mode], -1)


def _truncate_matrix_with_rank_limit(matrix: torch.Tensor, rank_limit: Optional[int]) -> torch.Tensor:
    if rank_limit is None or rank_limit <= 0:
        return matrix
    max_rank = min(matrix.shape)
    r = min(rank_limit, max_rank)
    if r <= 0 or max_rank == 0:
        return matrix
    try:
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    except RuntimeError:
        matrix = matrix + 1e-8 * torch.randn_like(matrix)
        U, S, Vh = torch.linalg.svd(matrix, full_matrices=False)
    U_r = U[:, :r]
    S_r = S[:r]
    Vh_r = Vh[:r, :]
    return (U_r * S_r.unsqueeze(0)) @ Vh_r


def _first_mode_unfold(tensor: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    shape = tuple(tensor.shape)
    if tensor.dim() == 1:
        matrix = tensor.reshape(tensor.shape[0], 1)
    else:
        matrix = tensor.reshape(tensor.shape[0], -1)
    return matrix, shape


def compute_pca_principal_angles(tensor_a: Optional[torch.Tensor], tensor_b: Optional[torch.Tensor], k: int) -> List[float]:
    if tensor_a is None or tensor_b is None:
        return []
    if tensor_a.shape != tensor_b.shape:
        return []
    if k <= 0:
        return []
    device = tensor_a.device
    dtype = torch.float32
    a_mat, _ = _first_mode_unfold(tensor_a.detach().to(device=device, dtype=dtype))
    b_mat, _ = _first_mode_unfold(tensor_b.detach().to(device=device, dtype=dtype))
    d, n = a_mat.shape
    max_rank = min(d, n)
    if max_rank == 0:
        return []
    k = min(k, max_rank)
    Ua, _, _ = torch.linalg.svd(a_mat, full_matrices=False)
    Ub, _, _ = torch.linalg.svd(b_mat, full_matrices=False)
    Ua_k = Ua[:, :k]
    Ub_k = Ub[:, :k]
    M = Ua_k.T @ Ub_k
    _, s, _ = torch.linalg.svd(M, full_matrices=False)
    s = torch.clamp(s, -1.0, 1.0)
    angles = torch.arccos(s) * (180.0 / math.pi)
    return angles.detach().cpu().tolist()


def compute_modewise_principal_angles(
    tensor_a: Optional[torch.Tensor],
    tensor_b: Optional[torch.Tensor],
    rank_products: Optional[List[int]] = None,
) -> List[float]:
    if tensor_a is None or tensor_b is None:
        return []
    if tensor_a.shape != tensor_b.shape:
        return []
    device = tensor_a.device
    target_dtype = torch.float32 if tensor_a.dtype != torch.float64 else tensor_a.dtype
    a = tensor_a.detach().to(device=device, dtype=target_dtype)
    b = tensor_b.detach().to(device=device, dtype=target_dtype)
    angles: List[float] = []
    for mode in range(a.dim()):
        unfold_a_matrix = _mode_unfold(a, mode)
        rank_limit = None
        if rank_products is not None and mode < len(rank_products):
            rank_limit = rank_products[mode]
        truncated_a = _truncate_matrix_with_rank_limit(unfold_a_matrix, rank_limit)
        unfold_a = truncated_a.reshape(-1)
        unfold_b = _mode_unfold(b, mode).reshape(-1)
        norm_a = torch.norm(unfold_a)
        norm_b = torch.norm(unfold_b)
        if norm_a <= 1e-12 or norm_b <= 1e-12:
            angles.append(float("nan"))
            continue
        cos_theta = torch.dot(unfold_a, unfold_b) / (norm_a * norm_b + 1e-12)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        angle_deg = torch.arccos(cos_theta) * (180.0 / math.pi)
        angles.append(angle_deg.item())
    return angles


def _load_image_tensor(path: str, resize: Optional[Tuple[int, int]], device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if resize is not None:
        image = image.resize(resize, Image.BILINEAR)
    arr = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).to(device=device, dtype=torch.float32)


def _mean_std_valid(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    valid = [v for v in values if v is not None and not math.isnan(v)]
    if not valid:
        return None, None
    return float(np.mean(valid)), float(np.std(valid))


def run_attack_angle_suite(
    X: torch.Tensor,
    config: EvalConfig,
    tr_rank: int,
    enable_adatr: bool,
    enable_fag: bool,
    enable_attr_attack: bool,
    pca_rank: int,
) -> Tuple[List[Dict[str, Optional[float]]], Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    X = X.to(dtype=torch.float32)
    device = X.device

    tr_cores_init = initialize_tr_cores_svd(X, tr_rank, device=str(device))

    results: List[Dict[str, Optional[float]]] = []
    recon_store: Dict[str, torch.Tensor] = {}
    noise_store: Dict[str, torch.Tensor] = {}
    attacked_store: Dict[str, torch.Tensor] = {}

    def add_result(label: str, E: torch.Tensor, recon: torch.Tensor):
        angle_vals = compute_pca_principal_angles(X, recon, pca_rank)
        ang_mean, ang_std = _mean_std_valid(angle_vals)
        results.append(
            {
                "method": label,
                "recon_error": torch.norm(X - recon).item(),
                "e_norm_f": torch.norm(E).item(),
                "e_norm_inf": torch.max(torch.abs(E)).item(),
                "angle_mean_deg": ang_mean,
                "angle_std_deg": ang_std,
            }
        )
        recon_store[label] = recon.detach()
        noise_store[label] = E.detach()
        attacked_store[label] = (X + E).detach()

    # baseline: clean TR reconstruction
    TN_clean = tl.tr_to_tensor(tr.tr_decompose2(X, clone_tr_cores(tr_cores_init), max_iter=config.inner_num_steps, tol=config.inner_tol, device=str(device)))
    zero_noise = torch.zeros_like(X)
    add_result("TR-Clean", zero_noise, TN_clean)

    # baseline: Gaussian noise with same energy as epsilon
    eps_val = config.epsilon
    if eps_val is None and config.epsilon_per_entry is not None:
        eps_val = config.epsilon_per_entry * X.numel()
    if eps_val is None:
        eps_val = 0.0
    if eps_val > 0:
        gaussian_noise = torch.randn_like(X)
        norm_sq = torch.norm(gaussian_noise).pow(2)
        if norm_sq > 0:
            scale = (eps_val / (norm_sq + 1e-12)) ** 0.5
            gaussian_noise = gaussian_noise * scale
        TN_gauss = tl.tr_to_tensor(tr.tr_decompose2(X + gaussian_noise, clone_tr_cores(tr_cores_init), max_iter=config.inner_num_steps, tol=config.inner_tol, device=str(device)))
        add_result("TR-Gaussian", gaussian_noise, TN_gauss)

    if enable_adatr:
        E_adatr, TN_adatr, _ = optimize_perturbation(
            X,
            clone_tr_cores(tr_cores_init),
            config,
        )
        add_result("AdaTR", E_adatr, TN_adatr)

    if enable_fag:
        E_fag, TN_fag, _ = optimize_perturbation_simplifed(
            X,
            clone_tr_cores(tr_cores_init),
            config,
            linf_bound=None,
        )
        add_result("FAG-AdaTR", E_fag, TN_fag)

    if enable_attr_attack:
        # 为 ATTR 攻击准备独立的配置，使用攻击外循环步数，并默认约束为全局 epsilon
        attr_eps = config.epsilon if config.epsilon is not None else config.attr_epsilon
        attr_cfg = EvalConfig(
            epsilon=config.epsilon,
            attack_lr=config.attack_lr,
            defence_lr=config.defence_lr,
            outer_num_steps_attack=config.outer_num_steps_attack,
            outer_num_steps_defence=config.outer_num_steps_attack,  # 攻击用相同步数
            inner_num_steps=config.inner_num_steps,
            inner_dis_num=config.inner_dis_num,
            inner_tol=config.inner_tol,
            outer_tol=config.outer_tol,
            attr_epsilon=attr_eps,
            epsilon_per_entry=config.epsilon_per_entry,
        )
        E_attr, TN_attr, _ = optimize_perturbation_comparison(
            X,
            clone_tr_cores(tr_cores_init),
            attr_cfg,
            attr_epsilon=attr_eps,
            clean_tensor=None,  # 作为攻击，不对干净样本做对齐
            linf_bound=None,
        )
        add_result("ATTR-Attack", E_attr, TN_attr)

    return results, recon_store, noise_store, attacked_store


def save_results_csv(rows: List[Dict[str, Optional[float]]], output_path: str):
    fieldnames = ["method", "recon_error", "e_norm_f", "e_norm_inf", "angle_mean_deg", "angle_std_deg"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_singular_values_csv(sv_dict: Dict[str, List[float]], output_path: str):
    # Normalize length by padding with empty cells
    max_len = max((len(v) for v in sv_dict.values()), default=0)
    methods = list(sv_dict.keys())
    rows = []
    for i in range(max_len):
        row = {}
        for m in methods:
            vals = sv_dict[m]
            row[m] = vals[i] if i < len(vals) else ""
        rows.append(row)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=methods)
        writer.writeheader()
        writer.writerows(rows)


def print_results_table(rows: List[Dict[str, Optional[float]]]):
    if not rows:
        print("No results to display.")
        return
    header = f"{'Method':<12}{'||X-T||':>15}{'||E||_F':>15}{'||E||_inf':>15}{'Angle Mean(deg)':>18}{'Angle Std(deg)':>17}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['method']:<12}"
            f"{row['recon_error']:>15.6f}"
            f"{row['e_norm_f']:>15.6f}"
            f"{row['e_norm_inf']:>15.6f}"
            f"{(row['angle_mean_deg'] if row['angle_mean_deg'] is not None else float('nan')):>18.4f}"
            f"{(row['angle_std_deg'] if row['angle_std_deg'] is not None else float('nan')):>17.4f}"
        )


def plot_singular_values(recon_store: Dict[str, torch.Tensor], output_path: str, top_k: int = 200):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    sv_dict: Dict[str, List[float]] = {}
    for label, tensor in recon_store.items():
        matrix, _ = _first_mode_unfold(tensor)
        # 使用 GPU 做奇异值分解
        matrix = matrix.to(device=tensor.device, dtype=torch.float32)
        try:
            sv = torch.linalg.svdvals(matrix)
        except RuntimeError:
            # 加微扰避免 SVD 错误
            matrix = matrix + 1e-6 * torch.randn_like(matrix)
            sv = torch.linalg.svdvals(matrix)
        sv = torch.sort(sv, descending=True).values
        if sv.numel() > top_k:
            sv = sv[:top_k]
        sv_cpu = sv.detach().cpu().numpy()
        sv_dict[label] = sv_cpu.tolist()
        plt.plot(sv_cpu, label=label)

    plt.xlabel("Index")
    plt.ylabel("Singular Value")
    plt.title("Singular Values of Reconstructions (mode-0 unfold)")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"奇异值曲线已保存到: {output_path}")
    return sv_dict


def plot_noise_singular_values_grid(
    noise_store: Dict[str, torch.Tensor],
    output_path: str,
    top_k: int = 200,
):
    import matplotlib.pyplot as plt

    labels = list(noise_store.keys())
    num = len(labels)
    if num == 0:
        return {}
    cols = 2
    rows = (num + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))
    axes = axes.flatten()
    sv_dict: Dict[str, List[float]] = {}
    for idx, label in enumerate(labels):
        ax = axes[idx]
        noise = noise_store[label]
        matrix, _ = _first_mode_unfold(noise)
        matrix = matrix.to(device=noise.device, dtype=torch.float32)
        try:
            sv = torch.linalg.svdvals(matrix)
        except RuntimeError:
            matrix = matrix + 1e-6 * torch.randn_like(matrix)
            sv = torch.linalg.svdvals(matrix)
        sv = torch.sort(sv, descending=True).values
        if sv.numel() > top_k:
            sv = sv[:top_k]
        sv_cpu = sv.detach().cpu().numpy()
        sv_dict[label] = sv_cpu.tolist()
        ax.plot(sv_cpu)
        ax.set_title(label)
        ax.set_yscale("log")
        ax.grid(True, linestyle="--", alpha=0.5)
    for j in range(idx + 1, rows * cols):
        fig.delaxes(axes[j])
    fig.suptitle("Noise Singular Values (mode-0 unfold)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"噪声奇异值子图已保存到: {output_path}")
    return sv_dict


def plot_downsample_distributions(
    noise_store: Dict[str, torch.Tensor],
    output_path: str,
    levels: int = 5,
    bins: int = 80,
):
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    labels_order = ["TR-Gaussian", "AdaTR", "FAG-AdaTR", "ATTR-Attack"]
    available = {k: v for k, v in noise_store.items() if k in labels_order}
    rows = len(labels_order)
    cols = levels
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if rows == 1:
        axes = axes[None, :]
    for r, label in enumerate(labels_order):
        noise = available.get(label)
        if noise is None:
            for c in range(cols):
                axes[r, c].axis("off")
            continue
        # 取单通道
        if noise.dim() == 3:
            # HWC -> CHW
            noise_chw = noise.permute(2, 0, 1)
        elif noise.dim() == 4:
            noise_chw = noise.squeeze(0)
        else:
            noise_chw = noise
        channel0 = noise_chw[0:1]  # shape (1,H,W)
        for c in range(cols):
            if c == 0:
                current = channel0
                title = "No DS"
            else:
                k = 2 ** c
                current = F.avg_pool2d(channel0, kernel_size=k, stride=k, ceil_mode=False)
                title = f"{k}x{k}"
            flat = current.detach().flatten().cpu().numpy()
            ax = axes[r, c]
            if flat.size == 0:
                ax.axis("off")
                continue
            ax.hist(flat, bins=bins, color="tab:blue", alpha=0.8)
            ax.set_title(f"{label} - {title}")
            ax.grid(True, linestyle="--", alpha=0.5)
    fig.suptitle("Downsampled Noise Distributions (channel 0)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"下采样分布图已保存到: {output_path}")


def plot_attacked_singular_values(attacked_store: Dict[str, torch.Tensor], output_path: str, top_k: int = 200):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    sv_dict: Dict[str, List[float]] = {}
    for label, tensor in attacked_store.items():
        matrix, _ = _first_mode_unfold(tensor)
        matrix = matrix.to(device=tensor.device, dtype=torch.float32)
        try:
            sv = torch.linalg.svdvals(matrix)
        except RuntimeError:
            matrix = matrix + 1e-6 * torch.randn_like(matrix)
            sv = torch.linalg.svdvals(matrix)
        sv = torch.sort(sv, descending=True).values
        if sv.numel() > top_k:
            sv = sv[:top_k]
        sv_cpu = sv.detach().cpu().numpy()
        sv_dict[label] = sv_cpu.tolist()
        plt.plot(sv_cpu, label=label)

    plt.xlabel("Index")
    plt.ylabel("Singular Value")
    plt.title("Singular Values of Attacked Tensors (X+E, mode-0 unfold)")
    plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"攻击张量奇异值曲线已保存到: {output_path}")
    return sv_dict


def _ensure_output_dir(path: str):
    directory = os.path.dirname(os.path.abspath(path))
    if directory == "":
        return
    try:
        os.makedirs(directory, exist_ok=True)
    except PermissionError:
        # 如果目录不可写，回退到当前工作目录
        fallback = os.path.join(os.getcwd(), os.path.basename(path))
        print(f"目录 {directory} 无写权限，自动改为 {fallback}")
        return fallback
    return path


def main():
    parser = argparse.ArgumentParser(description="Evaluate principal angles after adding adversarial perturbations and TR reconstruction.")
    parser.add_argument("--input-image", required=True, help="输入干净样本 (RGB 图片路径)。")
    parser.add_argument("--output-csv", required=True, help="结果表格输出路径。")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("W", "H"), default=None, help="可选：调整输入图像大小。")
    parser.add_argument("--epsilon-per-entry", type=float, default=5e-4, help="每个元素允许的均方噪声，最终 epsilon = value * numel.")
    parser.add_argument("--attr-epsilon", type=float, default=0.1, help="ATTR 攻击的 ||E||_F^2 约束。")
    parser.add_argument("--attack-lr", type=float, default=0.1, help="攻击学习率。")
    parser.add_argument("--defence-lr", type=float, default=0.01, help="防御学习率。")
    parser.add_argument("--outer-attack-steps", type=int, default=50, help="攻击外循环步数。")
    parser.add_argument("--outer-defence-steps", type=int, default=30, help="防御外循环步数（ATTR 用）。")
    parser.add_argument("--inner-steps", type=int, default=80, help="ALS 内循环步数。")
    parser.add_argument("--inner-dis-num", type=int, default=200, help="ALS 显示/间隔设置，传给 TR 分解。")
    parser.add_argument("--inner-tol", type=float, default=1e-8, help="ALS 内循环收敛阈值。")
    parser.add_argument("--outer-tol", type=float, default=1e-6, help="外循环收敛阈值。")
    parser.add_argument("--tr-rank", type=int, default=3, help="TR 初始秩（各模式统一）。")
    parser.add_argument("--pca-rank", type=int, default=10, help="PCA 子空间维度，用于计算主角。")
    parser.add_argument("--sv-plot", type=str, default=None, help="奇异值曲线输出路径（可选）。")
    parser.add_argument("--sv-topk", type=int, default=200, help="奇异值曲线保留的最大阶数。")
    parser.add_argument("--sv-plot-input", type=str, default=None, help="输入张量的奇异值曲线输出路径（可选）。")
    parser.add_argument("--sv-plot-attack", type=str, default=None, help="X+E 攻击张量奇异值曲线输出路径（可选）。")
    parser.add_argument("--sv-csv", type=str, default=None, help="奇异值表格输出路径（可选）。")
    parser.add_argument("--enable-downsample-exp", action="store_true", help="启用噪声下采样分布实验。")
    parser.add_argument("--downsample-plot", type=str, default=None, help="下采样分布 4x4 图输出路径（需要启用开关）。")
    parser.add_argument("--enable-adatr", action="store_true", default=True, help="启用 AdaTR 攻击。")
    parser.add_argument("--disable-adatr", dest="enable_adatr", action="store_false", help="禁用 AdaTR 攻击。")
    parser.add_argument("--enable-fag", action="store_true", default=True, help="启用 FAG-AdaTR 攻击。")
    parser.add_argument("--disable-fag", dest="enable_fag", action="store_false", help="禁用 FAG-AdaTR 攻击。")
    parser.add_argument("--enable-attr", action="store_true", default=True, help="启用 ATTR 攻击。")
    parser.add_argument("--disable-attr", dest="enable_attr", action="store_false", help="禁用 ATTR 攻击。")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = _load_image_tensor(args.input_image, tuple(args.resize) if args.resize else None, device)

    epsilon = args.epsilon_per_entry * X.numel()

    eval_config = EvalConfig(
        epsilon=epsilon,
        attack_lr=args.attack_lr,
        defence_lr=args.defence_lr,
        outer_num_steps_attack=args.outer_attack_steps,
        outer_num_steps_defence=args.outer_defence_steps,
        inner_num_steps=args.inner_steps,
        inner_dis_num=args.inner_dis_num,
        inner_tol=args.inner_tol,
        outer_tol=args.outer_tol,
        attr_epsilon=args.attr_epsilon,
        epsilon_per_entry=args.epsilon_per_entry,
    )

    rows, recon_store, noise_store, attacked_store = run_attack_angle_suite(
        X,
        eval_config,
        tr_rank=args.tr_rank,
        enable_adatr=args.enable_adatr,
        enable_fag=args.enable_fag,
        enable_attr_attack=args.enable_attr,
        pca_rank=args.pca_rank,
    )

    resolved_output = _ensure_output_dir(args.output_csv)
    output_path = resolved_output if resolved_output is not None else args.output_csv
    save_results_csv(rows, output_path)
    print_results_table(rows)
    print(f"结果已保存到: {output_path}")
    all_sv: Dict[str, List[float]] = {}
    # 输入张量奇异值
    if args.sv_plot_input:
        sv_input = plot_singular_values({"Input": X}, args.sv_plot_input, top_k=args.sv_topk)
        all_sv.update(sv_input)
    if args.sv_plot:
        sv_recon = plot_singular_values(recon_store, args.sv_plot, top_k=args.sv_topk)
        all_sv.update({f"Recon-{k}": v for k, v in sv_recon.items()})
    if args.sv_plot_input and noise_store:
        sv_noise = plot_noise_singular_values_grid(noise_store, args.sv_plot_input, top_k=args.sv_topk)
        all_sv.update({f"Noise-{k}": v for k, v in sv_noise.items()})
    if args.sv_plot_attack and attacked_store:
        sv_attack = plot_attacked_singular_values(attacked_store, args.sv_plot_attack, top_k=args.sv_topk)
        all_sv.update({f"Attack-{k}": v for k, v in sv_attack.items()})
    if args.sv_csv and all_sv:
        out_csv = _ensure_output_dir(args.sv_csv) or args.sv_csv
        save_singular_values_csv(all_sv, out_csv)
        print(f"奇异值表格已保存到: {out_csv}")
    if args.enable_downsample_exp and args.downsample_plot:
        plot_downsample_distributions(noise_store, args.downsample_plot)


if __name__ == "__main__":
    main()
