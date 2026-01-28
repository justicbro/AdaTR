import argparse
import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from collections import defaultdict
import tensorly as tl
import torch.autograd as autograd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import csv
from tensorly.decomposition import tensor_train, parafac, tucker
from tensorly.cp_tensor import CPTensor
import TR_func as tr
from attack_methods import (
    compute_linf_bound,
    objective_function_ATTR,
    optimize_perturbation,
    optimize_perturbation_comparison,
    optimize_perturbation_linf,
    optimize_perturbation_simplifed,
)
from datetime import datetime
import logging
import os
import glob
import math
from scipy.io import savemat
import time
import scipy.io
from contextlib import contextmanager

try:
    import psutil
except ImportError:
    psutil = None

# 指定使用第 0 张 GPU 卡
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

# 在代码中设置环境变量
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.autograd.set_detect_anomaly(True)

SEED = 66
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 测试是否启用了动态显存分配
if torch.cuda.is_available():
    print("Dynamic memory allocation enabled:")
    print(torch.cuda.memory_summary(device="cuda"))
else:
    print("CUDA is not available on this device.")

# 定义一个布尔变量来控制日志记录
log_enabled = True
imshow_enabled = False
save_mat_enabled = False
save_fig_enabled = False
plot_losses_enabled = True
distribution_plots_enabled = False
multi_decomposition_attack_enable = False

# Limit how many samples are used when plotting distributions to keep memory manageable
distribution_plot_sample_limit = 200000
singular_value_plot_limit = 300
multi_attack_global_stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"clean": [], "attack": [], "e_norm": []})
def _make_linf_stat_dict():
    return {"clean": [], "f_attack": [], "linf_attack": [], "f_E": [], "linf_E": []}

linf_attack_global_stats: Dict[str, Dict[str, List[float]]] = defaultdict(_make_linf_stat_dict)
current_peak_memory_recorder: Optional[Dict[str, Optional[int]]] = None
global_image_attack_summary: Dict[str, Dict[str, Optional[float]]] = defaultdict(dict)
global_case_order: List[str] = []


def format_bytes(num_bytes: Optional[float]) -> str:
    if num_bytes is None or num_bytes <= 0:
        return "N/A"
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(num_bytes)
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    return f"{value:.2f} {units[idx]}"


@contextmanager
def peak_memory_monitor(label: str):
    global current_peak_memory_recorder
    process = psutil.Process(os.getpid()) if psutil is not None else None
    start_rss = process.memory_info().rss if process is not None else None
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.empty_cache()
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    try:
        yield
    finally:
        peak_bytes: Optional[int] = None
        if cuda_available:
            try:
                peak_bytes = torch.cuda.max_memory_allocated()
            except Exception:
                peak_bytes = None
        elif process is not None and start_rss is not None:
            try:
                current_rss = process.memory_info().rss
                peak_bytes = max(0, current_rss - start_rss)
            except Exception:
                peak_bytes = None
        readable = format_bytes(peak_bytes)
        detail = readable if peak_bytes is None else f"{readable} ({int(peak_bytes)} bytes)"
        logging.info(f"[Peak Memory] {label}: {detail}")
        recorder = current_peak_memory_recorder
        if recorder is not None:
            recorder[label] = peak_bytes


def set_peak_memory_recorder(recorder: Optional[Dict[str, Optional[int]]]):
    global current_peak_memory_recorder
    current_peak_memory_recorder = recorder


def print_peak_memory_summary(records: Dict[str, Optional[int]]):
    if not records:
        return
    title = "Attack Peak Memory Usage"
    header = f"{'Attack':<20}{'Peak Memory':>18}"
    print(title)
    print(header)
    print('-' * len(header))
    logging.info(title)
    logging.info(header)
    logging.info('-' * len(header))
    for label, peak_bytes in records.items():
        readable = format_bytes(peak_bytes)
        line = f"{label:<20}{readable:>18}"
        print(line)
        logging.info(line)


def record_image_summary(image_name: str, rows: List[Dict[str, Any]]):
    global global_image_attack_summary, global_case_order
    if not rows:
        return
    for row in rows:
        label = row.get("label", "--")
        if label not in global_case_order:
            global_case_order.append(label)
        global_image_attack_summary[image_name][label] = row.get("tr_norm")


def print_global_image_summary():
    if not global_image_attack_summary or not global_case_order:
        return
    header = f"{'Image':<12}" + "".join(f"{case:>15}" for case in global_case_order)
    print("Per-Image Attack Summary (TR Norm)")
    print(header)
    print('-' * len(header))
    logging.info("Per-Image Attack Summary (TR Norm)")
    logging.info(header)
    logging.info('-' * len(header))
    for image, case_values in global_image_attack_summary.items():
        line = f"{image:<12}"
        for case in global_case_order:
            value = case_values.get(case)
            line += f"{value:>15.6f}" if value is not None else f"{'--':>15}"
        print(line)
        logging.info(line)


def _pca_subspace_basis_torch(matrix: torch.Tensor, k: int) -> torch.Tensor:
    if matrix.numel() == 0:
        return torch.zeros((matrix.shape[0], 0), device=matrix.device, dtype=matrix.dtype)
    U, _, _ = torch.linalg.svd(matrix, full_matrices=False)
    k = max(1, min(k, U.shape[1]))
    return U[:, :k]


def _asimov_distance_torch(U1: torch.Tensor, U2: torch.Tensor) -> torch.Tensor:
    if U1.shape[1] == 0 or U2.shape[1] == 0:
        return torch.zeros(1, device=U1.device, dtype=U1.dtype).squeeze()
    M = U1.T @ U2
    _, s, _ = torch.linalg.svd(M, full_matrices=False)
    s_min = torch.clamp(s.min(), -1.0, 1.0)
    return torch.arccos(s_min)


def _first_mode_unfold_tensor(tensor: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    shape = tuple(tensor.shape)
    if tensor.dim() == 1:
        matrix = tensor.reshape(tensor.shape[0], 1)
    else:
        matrix = tensor.reshape(tensor.shape[0], -1)
    return matrix, shape


def _fold_first_mode_tensor(matrix: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    return matrix.reshape(shape)


def run_pca_subspace_attack(
    tensor: torch.Tensor,
    epsilon: float,
    rank_k: int,
    n_trials_rank1: int,
    n_trials_full: int,
    mode: str = "rank1",
    verbose: bool = False,
    verbose_interval: int = 50,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    if epsilon is None or epsilon <= 0:
        zero = torch.zeros_like(tensor)
        return zero, tensor.clone(), 0.0

    with torch.no_grad():
        data_mat, original_shape = _first_mode_unfold_tensor(tensor.detach())
        d, n = data_mat.shape
        if d == 0 or n == 0:
            zero = torch.zeros_like(tensor)
            return zero, tensor.clone(), 0.0

        max_rank = min(d, n)
        if max_rank <= 1:
            zero = torch.zeros_like(tensor)
            return zero, tensor.clone(), 0.0
        rank_k = max(1, min(rank_k, max_rank - 1))
        U_orig = _pca_subspace_basis_torch(data_mat, rank_k)

        eta = math.sqrt(float(epsilon))
        best_theta = -1.0
        best_delta = None
        total_iters = 0
        interval = max(1, verbose_interval)

        def eval_delta(delta: torch.Tensor, label: str):
            nonlocal best_theta, best_delta, total_iters
            X_hat = data_mat + delta
            U_new = _pca_subspace_basis_torch(X_hat, rank_k)
            theta = _asimov_distance_torch(U_orig, U_new)
            theta_val = float(theta)
            total_iters += 1
            if verbose and total_iters % interval == 0:
                print(f"[PCA Attack][{label}] iter {total_iters}: theta={theta_val:.6f}, best={max(best_theta, 0):.6f}")
            if theta_val > best_theta:
                best_theta = theta_val
                best_delta = delta.clone()

        device = data_mat.device
        dtype = data_mat.dtype

        if mode in ("rank1", "both"):
            trials = max(1, n_trials_rank1)
            for _ in range(trials):
                a = torch.randn(d, device=device, dtype=dtype)
                b = torch.randn(n, device=device, dtype=dtype)
                na = torch.norm(a)
                nb = torch.norm(b)
                if na.item() == 0.0 or nb.item() == 0.0:
                    continue
                scale = eta / (na.item() * nb.item())
                delta = torch.outer(a * scale, b)
                eval_delta(delta, "rank1")

        if mode in ("full", "both"):
            trials = max(1, n_trials_full)
            for _ in range(trials):
                delta = torch.randn(d, n, device=device, dtype=dtype)
                fro = torch.norm(delta)
                if fro.item() == 0.0:
                    continue
                delta = delta * (eta / fro.item())
                eval_delta(delta, "full")

        if best_delta is None:
            zero = torch.zeros_like(tensor)
            return zero, tensor.clone(), 0.0

        delta_tensor = _fold_first_mode_tensor(best_delta, original_shape).to(device=tensor.device, dtype=tensor.dtype)
        attacked = tensor + delta_tensor
        return delta_tensor, attacked, float(max(best_theta, 0.0))

# LR Finder settings
lr_finder_enabled = False
lr_finder_min_lr = 1e-7
lr_finder_max_lr = 1e-1
lr_finder_num_points = 20
lr_finder_folder_name = 'lr_finder'

# Attack/defence switches
attack_enable_clean = True
attack_enable_gaussian = True
attack_enable_ours_f = True
attack_enable_ours = True
attack_enable_attr_attack = True
attack_enable_pca_subspace = False
attack_multi_tr_enable = True
attack_multi_tt_enable = True
attack_multi_tucker_enable = True
attack_multi_cp_enable = True
linf_attack_enable = False
attr_defence_enable = False
attr_dfadatr_enable = False
df_adatr_enable = False
df_adatr_noisy_enable = False
ata_defence_enable = False
adveresarial_training_enable = False
theta_defence_enable = False
tt_attack_rank = 20
tucker_attack_rank = 20
cp_attack_rank = 22
multi_attack_inner_iters = 100
pca_attack_rank = 3
pca_attack_rank1_trials = 300
pca_attack_full_trials = 300
pca_attack_mode = "rank1"  # choices: rank1, full, both
pca_attack_verbose = False
pca_attack_verbose_interval = 50

# 创建日志文件夹
if log_enabled:
    log_folder = 'Defence Results My picture7'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# 配置日志
    log_filename = os.path.join(log_folder, f'AT_TR_compare_{current_time}.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

tl.set_backend('pytorch')  # Set tensorly to use PyTorch backend

class OptimizationConfig:
    def __init__(
        self,
        epsilon,
        lr,
        outer_num_steps_attack,
        outer_num_steps_defence,
        inner_num_steps,
        outer_dis_num,
        inner_dis_num,
        inner_tol,
        attr_epsilon=None,
        unroll_steps=None,
        attack_lr=None,
        defence_lr=None,
        epsilon_per_entry=None,
        theta_epsilon=None,
        outer_tol=1e-6,
    ):
        self.epsilon = epsilon
        self.lr = lr
        self.outer_num_steps_attack = outer_num_steps_attack
        self.outer_num_steps_defence = outer_num_steps_defence
        self.inner_num_steps = inner_num_steps
        self.outer_dis_num = outer_dis_num
        self.inner_dis_num = inner_dis_num
        self.inner_tol = inner_tol
        self.attr_epsilon = attr_epsilon
        self.unroll_steps = unroll_steps
        self.attack_lr = attack_lr if attack_lr is not None else lr
        self.defence_lr = defence_lr if defence_lr is not None else lr
        self.epsilon_per_entry = epsilon_per_entry
        self.theta_epsilon = theta_epsilon
        self.outer_tol = outer_tol

def clone_config(cfg: OptimizationConfig) -> OptimizationConfig:
    return OptimizationConfig(
        cfg.epsilon,
        cfg.lr,
        cfg.outer_num_steps_attack,
        cfg.outer_num_steps_defence,
        cfg.inner_num_steps,
        cfg.outer_dis_num,
        cfg.inner_dis_num,
        cfg.inner_tol,
        attr_epsilon=cfg.attr_epsilon,
        unroll_steps=cfg.unroll_steps,
        attack_lr=cfg.attack_lr,
        defence_lr=cfg.defence_lr,
        epsilon_per_entry=cfg.epsilon_per_entry,
        theta_epsilon=cfg.theta_epsilon,
        outer_tol=cfg.outer_tol,
    )


def scaled_config_for_tensor(cfg: OptimizationConfig, tensor: torch.Tensor) -> OptimizationConfig:
    scaled_cfg = clone_config(cfg)
    if cfg.epsilon_per_entry is not None:
        epsilon_val = float(cfg.epsilon_per_entry) * tensor.numel()
        scaled_cfg.epsilon = epsilon_val
        if scaled_cfg.theta_epsilon is None:
            scaled_cfg.theta_epsilon = epsilon_val
    return scaled_cfg

# 定义保存图像的函数
def save_fig(output_folder, base_name, fig):
    if save_fig_enabled:
        # 确保输出文件夹存在
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # 保存为 PNG 格式
        fig.savefig(os.path.join(output_folder, f'{base_name}.png'), bbox_inches='tight', pad_inches=0)
        # 保存为 JPEG 格式
        fig.savefig(os.path.join(output_folder, f'{base_name}.jpeg'), format='jpeg', bbox_inches='tight', pad_inches=0)
        print(f"Figures saved: {os.path.join(output_folder, f'{base_name}.png')} and {os.path.join(output_folder, f'{base_name}.jpeg')}")


def save_mat_file(output_folder, variable_name, data):
    if save_mat_enabled:
        # 确保输出文件夹存在
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        # 保存 .mat 文件
        savemat(os.path.join(output_folder, f'{variable_name}.mat'), {variable_name: data})
        print(f".mat file saved: {os.path.join(output_folder, f'{variable_name}.mat')}")

def display_line_graphs(loss_dict, figsize=(10, 5), save_path=None):
    plt.figure(figsize=figsize)
    for label, loss_values in loss_dict.items():
        plt.plot(loss_values, label=label)
    plt.xlabel('Step')
    plt.ylabel('Loss (sqrt)')
    plt.title('Loss over Steps')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_loss_curves(curves_dict, title, ylabel, save_path, baseline=None):
    if not curves_dict and baseline is None:
        return
    plt.figure(figsize=(10, 6))
    has_curve = False
    for label, losses in curves_dict.items():
        if not losses:
            continue
        steps = list(range(1, len(losses) + 1))
        plt.plot(steps, losses, label=label)
        has_curve = True
    if baseline is not None:
        plt.axhline(baseline, color='gray', linestyle='--', label='Baseline')
    if not has_curve and baseline is None:
        plt.close()
        return
    plt.xlabel('Step')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def clone_tr_cores(cores: List[torch.Tensor]) -> List[torch.Tensor]:
    return [core.clone().detach() for core in cores]


def load_video_tensor(
    video_path: str,
    min_frames: int = 20,
    max_frames: int = 30,
    target_size: Optional[Tuple[int, int]] = None,
    downscale_factor: float = 0.5,
) -> torch.Tensor:
    data = scipy.io.loadmat(video_path)
    video_array = None
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.ndim == 4:
            video_array = value
            break
    if video_array is None:
        raise ValueError(f"No 4-D video tensor found in {video_path}")

    video_array = np.array(video_array)
    if video_array.dtype != np.float32:
        video_array = video_array.astype(np.float32)
    if video_array.max() > 1.0:
        video_array = video_array / 255.0

    H, W = video_array.shape[0], video_array.shape[1]

    def resize_video(array: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        resized = np.empty((out_h, out_w, array.shape[2], array.shape[3]), dtype=np.float32)
        for t in range(array.shape[3]):
            frame = array[..., t]
            frame_uint8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            pil_img = Image.fromarray(frame_uint8)
            pil_img = pil_img.resize((out_w, out_h), Image.BILINEAR)
            resized[..., t] = np.asarray(pil_img, dtype=np.float32) / 255.0
        return resized

    if target_size is not None:
        target_w, target_h = target_size
        target_h = max(1, int(target_h))
        target_w = max(1, int(target_w))
        if target_h != H or target_w != W:
            video_array = resize_video(video_array, target_h, target_w)
    else:
        new_H = max(1, int(round(H * downscale_factor)))
        new_W = max(1, int(round(W * downscale_factor)))
        if new_H != H or new_W != W:
            video_array = resize_video(video_array, new_H, new_W)

    total_frames = video_array.shape[-1]
    if total_frames < min_frames:
        raise ValueError(
            f"Video {video_path} has only {total_frames} frames (<{min_frames})"
        )
    if total_frames > max_frames:
        logging.info(
            f"Video {video_path} has {total_frames} frames; truncating to first {max_frames} frames."
        )
        video_array = video_array[..., :max_frames]

    return torch.from_numpy(video_array)


def initialize_tr_cores_svd(
    X: torch.Tensor,
    ranks: List[int],
    device: str = "cuda",
    max_iter: int = 10,
    tol: float = 1e-6,
) -> List[torch.Tensor]:
    shape = list(X.shape)
    cores = tr.initialize_tr_cores(shape, ranks, device=device)
    X_detached = X.detach()
    cores = tr.tr_decompose2(
        X_detached,
        cores,
        max_iter=max_iter,
        tol=tol,
        device=device,
    )
    return [core.detach() for core in cores]


def plot_loss_curves(curves_dict, title, ylabel, save_path, baseline=None):
    if not curves_dict:
        return
    plt.figure(figsize=(10, 6))
    has_valid = False
    for label, losses in curves_dict.items():
        if not losses:
            continue
        steps = list(range(1, len(losses) + 1))
        plt.plot(steps, losses, label=label)
        has_valid = True
    if not has_valid and baseline is None:
        plt.close()
        return
    if baseline is not None:
        plt.axhline(baseline, color='gray', linestyle='--', label='Baseline')
    plt.xlabel('Step')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def make_grad_printer(name):
    def _hook(grad):
        has_nan = torch.isnan(grad).any().item()
        has_inf = torch.isinf(grad).any().item()
        print(
            f"{name} grad | "
            f"norm={grad.norm():.4e} | "
            f"min={grad.min():.4e} | "
            f"max={grad.max():.4e} | "
            f"nan={has_nan} | inf={has_inf}"
        )
    return _hook

def hook_fn(grad):
    print("Gradient for E:", grad)

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Assuming the image is normalized to [0, 1]
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def display_images(original_image, images, row_titles, col_titles, rows, cols, figsize=(20, 10), save_path=None):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    # 添加行标题
    for row, ax in enumerate(axes[:, 0]):  # 第一列的每个轴
        ax.annotate(
            row_titles[row], xy=(0, 0.5), xycoords='axes fraction',
            textcoords='offset points', size='xx-large', ha='right', va='center',
            rotation=0, xytext=(-50, 0), annotation_clip=False
        )

    # 调整每个列标题的位置，使其精确对齐
    for col, ax in enumerate(axes[0, :]):
        col_center = ax.get_position().x0 + ax.get_position().width / 2
        fig.text(
            col_center, 0.95,  # 动态计算每列的中心位置
            col_titles[col], ha='center', va='bottom', size='xx-large'
        )

    for i, ax in enumerate(axes.flat):
        row = i // cols
        col = i % cols
        ax.imshow(images[i])
        psnr = calculate_psnr(original_image, images[i])
        ax.set_title(f"\nPSNR: {psnr:.2f} dB")
        ax.axis('off')

    if save_path:
        plt.savefig(save_path)
    plt.show()


def tensor_to_numpy_array(tensor):
    if tensor is None:
        return None
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().float().numpy()
    if isinstance(tensor, np.ndarray):
        return tensor
    return np.array(tensor, dtype=np.float32)


def flatten_for_histogram(array, max_samples=distribution_plot_sample_limit):
    if array is None:
        return None
    flat = np.asarray(array).ravel()
    if flat.size == 0:
        return None
    finite_mask = np.isfinite(flat)
    flat = flat[finite_mask]
    if flat.size == 0:
        return None
    if max_samples and flat.size > max_samples:
        indices = np.random.choice(flat.size, size=max_samples, replace=False)
        flat = flat[indices]
    return flat


def fft_magnitude(array):
    if array is None:
        return None
    fft_values = np.fft.fftn(array)
    return np.abs(fft_values)


def _plot_distribution_grid(series, title, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    for ax, (label, data), color in zip(axes.flat, series, colors):
        values = flatten_for_histogram(data)
        if values is None or values.size == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
            ax.set_title(f"{label} (empty)")
            ax.axis('off')
            continue
        ax.hist(values, bins=80, color=color, alpha=0.85)
        ax.set_title(label)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

def _prepare_hw_channels(array):
    if array is None:
        return None
    arr = np.asarray(array)
    if arr.ndim < 2:
        return None
    if arr.ndim == 2:
        arr = arr[..., None]
    elif arr.ndim > 3:
        h, w = arr.shape[0], arr.shape[1]
        remaining = int(np.prod(arr.shape[2:]))
        arr = arr.reshape(h, w, remaining)
    elif arr.shape[2] == 0:
        return None
    return arr

def compute_channel_singular_values(array):
    arr = _prepare_hw_channels(array)
    if arr is None:
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    sv_list = []
    for idx in range(arr.shape[2]):
        channel = arr[..., idx]
        try:
            channel_tensor = torch.as_tensor(channel, dtype=dtype, device=device)
            singular_vals = torch.linalg.svdvals(channel_tensor)
        except RuntimeError:
            noise = 1e-8 * torch.randn(channel.shape, dtype=dtype, device=device)
            channel_tensor = torch.as_tensor(channel, dtype=dtype, device=device) + noise
            singular_vals = torch.linalg.svdvals(channel_tensor)
        singular_vals = singular_vals.detach().cpu().numpy()
        if singular_vals.size == 0:
            continue
        sv_list.append(singular_vals)
    return sv_list if sv_list else None


def _plot_singular_value_grid(series, title, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    color_cycle = plt.cm.get_cmap('tab10')
    for ax, (label, sv_list) in zip(axes.flat, series):
        if not sv_list:
            ax.text(0.5, 0.5, 'No singular values', ha='center', va='center', fontsize=12)
            ax.set_title(f"{label} (empty)")
            ax.axis('off')
            continue
        has_curve = False
        for idx, singular_vals in enumerate(sv_list):
            if singular_vals is None or singular_vals.size == 0:
                continue
            trimmed = singular_vals[:singular_value_plot_limit]
            color = color_cycle(idx % 10)
            ax.plot(trimmed, label=f'Channel {idx + 1}', color=color)
            has_curve = True
        ax.set_title(label)
        if has_curve:
            ax.set_xlabel('Index')
            ax.set_ylabel('Singular Value')
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No finite values', ha='center', va='center', fontsize=12)
            ax.axis('off')
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


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
            angles.append(float('nan'))
            continue
        cos_theta = torch.dot(unfold_a, unfold_b) / (norm_a * norm_b + 1e-12)
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        angle_deg = torch.arccos(cos_theta) * (180.0 / math.pi)
        angles.append(angle_deg.item())
    return angles


def _mean_angle(angle_values: List[float]) -> Optional[float]:
    valid = [val for val in angle_values if val is not None and not math.isnan(val)]
    if not valid:
        return None
    return float(np.mean(valid))


def print_principal_angle_table(base_name: str, angle_summary: Dict[str, Dict[str, Optional[float]]]):
    if not angle_summary:
        return
    header = f"{'Attack':<18}{'Angle(E,Xhat)':>18}{'Angle(E,Recon)':>18}"
    separator = '-' * len(header)
    title = f"Principal Angles Summary ({base_name})"
    print(title)
    print(header)
    print(separator)
    logging.info(title)
    logging.info(header)
    logging.info(separator)
    for label, metrics in angle_summary.items():
        clean_mean = metrics.get("clean_mean")
        recon_mean = metrics.get("attack_mean")
        clean_str = f"{clean_mean:.4f}" if clean_mean is not None else "--"
        recon_str = f"{recon_mean:.4f}" if recon_mean is not None else "--"
        line = f"{label:<18}{clean_str:>18}{recon_str:>18}"
        print(line)
        logging.info(line)


def _expand_rank_list(rank_value: Union[int, List[int]], order: int, max_dims: Sequence[int]) -> List[int]:
    if isinstance(rank_value, int):
        return [min(rank_value, int(max_dims[i])) for i in range(order)]
    if isinstance(rank_value, (list, tuple)):
        if len(rank_value) == order:
            return [min(int(rank_value[i]), int(max_dims[i])) for i in range(order)]
        raise ValueError(f"Rank list length {len(rank_value)} must match tensor order {order}.")
    raise ValueError("Rank value must be int or list/tuple.")


def _expand_tt_rank(rank_value: Union[int, List[int]], order: int) -> List[int]:
    if isinstance(rank_value, int):
        ranks = [1] + [rank_value] * (order - 1) + [1]
        return ranks
    if isinstance(rank_value, (list, tuple)):
        if len(rank_value) == order + 1:
            return list(rank_value)
        raise ValueError(f"TT rank list length must be order+1 ({order + 1}).")
    raise ValueError("TT rank must be int or list/tuple.")


def reconstruct_tensor_with_method(
    tensor: torch.Tensor,
    method: str,
    config: OptimizationConfig,
    *,
    tr_core_init: Optional[List[torch.Tensor]] = None,
    tt_rank: Optional[Union[int, List[int]]] = None,
    tucker_rank: Optional[Union[int, List[int]]] = None,
    cp_rank: Optional[int] = None,
    inner_num_steps: Optional[int] = None,
) -> Tuple[torch.Tensor, Any]:
    method_lower = method.lower()
    iterations = inner_num_steps if inner_num_steps is not None else config.inner_num_steps
    tensor = tensor.to(dtype=torch.float32)
    if method_lower == "tr":
        if tr_core_init is None:
            raise ValueError("TR reconstruction requires initial cores.")
        working_cores = tr.tr_decompose2(
            tensor,
            clone_tr_cores(tr_core_init),
            max_iter=iterations,
            tol=config.inner_tol,
            dis_num=config.inner_dis_num,
            device=tensor.device,
        )
        recon = tl.tr_to_tensor(working_cores)
        return recon, working_cores
    if method_lower == "tt":
        ranks = _expand_tt_rank(tt_rank if tt_rank is not None else 3, tensor.dim())
        tt_cores = tensor_train(tensor, rank=ranks, verbose=False)
        recon = tl.tr_to_tensor(tt_cores)
        return recon, tt_cores
    if method_lower == "tucker":
        rank_list = _expand_rank_list(
            tucker_rank if tucker_rank is not None else 3,
            tensor.dim(),
            tensor.shape,
        )
        core, factors = tucker(
            tensor,
            rank=rank_list,
            n_iter_max=iterations,
            init="svd",
            tol=config.inner_tol,
        )
        recon = tl.tucker_to_tensor((core, factors))
        return recon, (core, factors)
    if method_lower == "cp":
        rank_val = cp_rank if cp_rank is not None else 5
        try:
            cp_tensor = parafac(
                tensor,
                rank=rank_val,
                n_iter_max=iterations,
                init="svd",
                tol=config.inner_tol,
                normalize_factors=False,
            )
            recon = tl.cp_to_tensor(cp_tensor)
            weights = cp_tensor.weights if cp_tensor.weights is not None else None
            weights = weights.to(tensor.device) if weights is not None else None
            factors = [factor.to(tensor.device) for factor in cp_tensor.factors]
            cp_tensor = CPTensor((weights, factors))
            return recon, cp_tensor
        except RuntimeError as exc:
            if "CUDA out of memory" not in str(exc):
                raise
            logging.warning("[CP] CUDA OOM encountered, reducing rank and retrying.")
            print("[CP] CUDA OOM encountered, reducing rank and retrying.")
            torch.cuda.empty_cache()
            reduced_rank = max(rank_val // 2, 1)
            if reduced_rank == rank_val:
                raise
            cp_tensor = parafac(
                tensor,
                rank=reduced_rank,
                n_iter_max=iterations,
                init="svd",
                tol=config.inner_tol,
                normalize_factors=False,
            )
            recon = tl.cp_to_tensor(cp_tensor)
            weights = cp_tensor.weights if cp_tensor.weights is not None else None
            weights = weights.to(tensor.device) if weights is not None else None
            factors = [factor.to(tensor.device) for factor in cp_tensor.factors]
            cp_tensor = CPTensor((weights, factors))
            return recon, cp_tensor
    raise ValueError(f"Unknown decomposition method: {method}")


def optimize_als_attack(
    X: torch.Tensor,
    config: OptimizationConfig,
    method: str,
    *,
    tr_core_init: Optional[List[torch.Tensor]] = None,
    tt_rank: Optional[Union[int, List[int]]] = None,
    tucker_rank: Optional[Union[int, List[int]]] = None,
    cp_rank: Optional[int] = None,
    inner_num_steps: Optional[int] = None,
    label: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[float], float]:
    if inner_num_steps is None:
        inner_num_steps = config.inner_num_steps

    attack_lr = getattr(config, "attack_lr", getattr(config, "lr", 0.01))
    E = torch.randn_like(X, device=X.device, requires_grad=True)
    scaling_factor = (config.epsilon ** 0.5) / torch.norm(E)
    if torch.norm(E) ** 2 > config.epsilon:
        E.data = scaling_factor * E

    optimizer = torch.optim.Adam([E], lr=attack_lr)
    loss_history: List[float] = []

    for step in range(config.outer_num_steps_attack):
        optimizer.zero_grad()

        attacked_tensor = X + E
        try:
            recon, _ = reconstruct_tensor_with_method(
                attacked_tensor,
                method,
                config,
                tr_core_init=tr_core_init,
                tt_rank=tt_rank,
                tucker_rank=tucker_rank,
                cp_rank=cp_rank,
                inner_num_steps=inner_num_steps,
            )
        except Exception as exc:
            logging.exception(f"[{method}] Decomposition failed at step {step + 1}: {exc}")
            print(f"[{method}] Decomposition failed at step {step + 1}: {exc}")
            break

        loss = -torch.norm(X - recon) ** 2
        if torch.isnan(loss):
            print(f"[{method}] Loss is NaN at step {step + 1}, stopping.")
            break

        loss.backward()
        optimizer.step()
        if linf_bound is not None:
            with torch.no_grad():
                E.data.clamp_(-linf_bound, linf_bound)

        scaling_factor = (config.epsilon ** 0.5) / torch.norm(E)
        if torch.norm(E) ** 2 > config.epsilon:
            E.data = scaling_factor * E

        recon_error = torch.norm(X - recon).item()
        loss_history.append(recon_error)
        label_print = label if label is not None else method
        logging.info(
            f"[{label_print}] Step {step + 1}/{config.outer_num_steps_attack}, recon_err={recon_error:.6f}, ||E||={torch.norm(E).item():.6f}"
        )

    with torch.no_grad():
        final_recon, _ = reconstruct_tensor_with_method(
            X + E,
            method,
            config,
            tr_core_init=tr_core_init,
            tt_rank=tt_rank,
            tucker_rank=tucker_rank,
            cp_rank=cp_rank,
            inner_num_steps=inner_num_steps,
        )
        final_e_norm = torch.norm(E).item()
    return E.detach(), final_recon.detach(), loss_history, final_e_norm


def print_linf_attack_table(base_name: str, rows: List[Dict[str, Optional[float]]]):
    if not rows:
        return
    title = f"L2 vs L∞ AdaTR Comparison ({base_name})"
    header = f"{'Case':<18}{'||X-T||':>15}{'||E||':>15}"
    print(title)
    print(header)
    print('-' * len(header))
    logging.info(title)
    logging.info(header)
    logging.info('-' * len(header))
    for row in rows:
        recon = row.get("recon")
        e_norm = row.get("e_norm")
        recon_str = f"{recon:.6f}" if recon is not None else "--"
        e_str = f"{e_norm:.6f}" if e_norm is not None else "--"
        line = f"{row.get('case','--'):<18}{recon_str:>15}{e_str:>15}"
        print(line)
        logging.info(line)


def record_linf_attack_results(method: str, clean: Optional[float], f_attack: Optional[float], linf_attack: Optional[float], linf_e: Optional[float], f_e: Optional[float] = None):
    stats = linf_attack_global_stats[method]
    if clean is not None and not math.isnan(clean):
        stats["clean"].append(clean)
    if f_attack is not None and not math.isnan(f_attack):
        stats["f_attack"].append(f_attack)
    if linf_attack is not None and not math.isnan(linf_attack):
        stats["linf_attack"].append(linf_attack)
    if f_e is not None and not math.isnan(f_e):
        stats["f_E"].append(f_e)
    if linf_e is not None and not math.isnan(linf_e):
        stats["linf_E"].append(linf_e)


def run_linf_attack_experiment(
    X: torch.Tensor,
    base_name: str,
    config: OptimizationConfig,
    tr_cores_init: List[torch.Tensor],
    clean_norm: Optional[float],
    clean_reconstruction: Optional[torch.Tensor],
) -> Dict[str, Optional[float]]:
    # Ensure clean baseline
    if clean_norm is None or clean_reconstruction is None:
        tr_cores_clean = tr.tr_decompose2(X, clone_tr_cores(tr_cores_init), max_iter=config.inner_num_steps, tol=config.inner_tol, device="cuda")
        clean_reconstruction = tl.tr_to_tensor(tr_cores_clean)
        clean_norm = torch.norm(X - clean_reconstruction).item()

    linf_bound = compute_linf_bound(config, X)
    rows = [{"case": "Clean TR", "recon": clean_norm, "e_norm": None}]
    summary: Dict[str, Dict[str, Optional[float]]] = {}

    def add_method_row(label, f_recon, f_e, linf_recon, linf_e):
        rows.append({"case": f"{label} (F)", "recon": f_recon, "e_norm": f_e})
        rows.append({"case": f"{label} (L∞)", "recon": linf_recon, "e_norm": linf_e})
        record_linf_attack_results(label, clean_norm, f_recon, linf_recon, linf_e, f_e)
        summary[label] = {
            "clean": clean_norm,
            "f_attack": f_recon,
            "linf_attack": linf_recon,
            "linf_E": linf_e,
            "E_f": f_e,
        }

    # AdaTR (F)
    with peak_memory_monitor("AdaTR_F"):
        E_adatr_f, TN_adatr_f, _ = optimize_perturbation(
            X,
            clone_tr_cores(tr_cores_init),
            config,
        )
    adatr_f_recon = torch.norm(X - TN_adatr_f).item()
    adatr_f_e = torch.norm(E_adatr_f).item()
    # AdaTR (Linf)
    with peak_memory_monitor("AdaTR_Linf"):
        E_adatr_linf, TN_adatr_linf, _ = optimize_perturbation_linf(
            X,
            clone_tr_cores(tr_cores_init),
            config,
            linf_bound,
        )
    adatr_linf_recon = torch.norm(X - TN_adatr_linf).item()
    adatr_linf_e = torch.max(torch.abs(E_adatr_linf)).item()
    add_method_row("AdaTR", adatr_f_recon, adatr_f_e, adatr_linf_recon, adatr_linf_e)

    # FAG-AdaTR
    E_fag_f, TN_fag_f, _ = optimize_perturbation_simplifed(
        X,
        clone_tr_cores(tr_cores_init),
        config,
        linf_bound=None,
    )
    fag_f_recon = torch.norm(X - TN_fag_f).item()
    fag_f_e = torch.norm(E_fag_f).item()
    E_fag_linf, TN_fag_linf, _ = optimize_perturbation_simplifed(
        X,
        clone_tr_cores(tr_cores_init),
        config,
        linf_bound=linf_bound,
    )
    fag_linf_recon = torch.norm(X - TN_fag_linf).item()
    fag_linf_e = torch.max(torch.abs(E_fag_linf)).item()
    add_method_row("FAG-AdaTR", fag_f_recon, fag_f_e, fag_linf_recon, fag_linf_e)

    # ATTR
    attr_cfg = clone_config(config)
    attr_cfg.attr_epsilon = config.epsilon
    attr_eps = attr_cfg.attr_epsilon if attr_cfg.attr_epsilon is not None else attr_cfg.epsilon
    E_attr_f, TN_attr_f, _ = optimize_perturbation_comparison(
        X.detach(),
        clone_tr_cores(tr_cores_init),
        attr_cfg,
        attr_epsilon=attr_eps,
        clean_tensor=None,
        linf_bound=None,
    )
    attr_f_recon = torch.norm(X - TN_attr_f).item()
    attr_f_e = torch.norm(E_attr_f).item()
    E_attr_linf, TN_attr_linf, _ = optimize_perturbation_comparison(
        X.detach(),
        clone_tr_cores(tr_cores_init),
        attr_cfg,
        attr_epsilon=attr_eps,
        clean_tensor=None,
        linf_bound=linf_bound,
    )
    attr_linf_recon = torch.norm(X - TN_attr_linf).item()
    attr_linf_e = torch.max(torch.abs(E_attr_linf)).item()
    add_method_row("ATTR", attr_f_recon, attr_f_e, attr_linf_recon, attr_linf_e)

    print_linf_attack_table(base_name, rows)
    summary["clean"] = {"clean": clean_norm}
    return summary


def print_multi_attack_results(base_name: str, rows: List[Dict[str, Optional[float]]]):
    if not rows:
        return
    title = f"Multi-Decomposition Attack Summary ({base_name})"
    header = f"{'Method':<12}{'||X-T|| Clean':>18}{'||X-T|| Attack':>18}{'||E||':>15}"
    print(title)
    print(header)
    print('-' * len(header))
    logging.info(title)
    logging.info(header)
    logging.info('-' * len(header))
    for row in rows:
        clean_val = row.get("clean_error")
        attack_val = row.get("attack_error")
        e_norm = row.get("e_norm")
        clean_str = f"{clean_val:.6f}" if clean_val is not None else "--"
        attack_str = f"{attack_val:.6f}" if attack_val is not None else "--"
        e_str = f"{e_norm:.6f}" if e_norm is not None else "--"
        line = f"{row.get('method','--'):<12}{clean_str:>18}{attack_str:>18}{e_str:>15}"
        print(line)
        logging.info(line)


def record_multi_attack_results(base_name: str, rows: List[Dict[str, Optional[float]]]):
    if not rows:
        return
    for row in rows:
        method = row.get("method")
        if not method:
            continue
        clean_val = row.get("clean_error")
        attack_val = row.get("attack_error")
        if clean_val is not None and not math.isnan(clean_val):
            multi_attack_global_stats[method]["clean"].append(clean_val)
        if attack_val is not None and not math.isnan(attack_val):
            multi_attack_global_stats[method]["attack"].append(attack_val)
        e_norm = row.get("e_norm")
        if e_norm is not None and not math.isnan(e_norm):
            multi_attack_global_stats[method]["e_norm"].append(e_norm)


def print_multi_attack_global_summary(output_folder: str):
    if not multi_attack_global_stats:
        return
    title = "Aggregated Multi-Decomposition Attack Summary"
    header = f"{'Method':<12}{'Clean Mean±Std':>22}{'Attack Mean±Std':>24}{'||E|| Mean±Std':>24}"
    print(title)
    print(header)
    print('-' * len(header))
    logging.info(title)
    logging.info(header)
    logging.info('-' * len(header))
    table_rows = []
    for method, values in multi_attack_global_stats.items():
        clean_vals = values.get("clean", [])
        attack_vals = values.get("attack", [])
        e_vals = values.get("e_norm", [])
        clean_mean = np.mean(clean_vals) if clean_vals else None
        clean_std = np.std(clean_vals) if clean_vals else None
        attack_mean = np.mean(attack_vals) if attack_vals else None
        attack_std = np.std(attack_vals) if attack_vals else None
        e_mean = np.mean(e_vals) if e_vals else None
        e_std = np.std(e_vals) if e_vals else None
        clean_str = "--"
        attack_str = "--"
        e_str = "--"
        if clean_mean is not None:
            clean_str = f"{clean_mean:.6f}±{clean_std:.6f}" if clean_std is not None else f"{clean_mean:.6f}"
        if attack_mean is not None:
            attack_str = f"{attack_mean:.6f}±{attack_std:.6f}" if attack_std is not None else f"{attack_mean:.6f}"
        if e_mean is not None:
            e_str = f"{e_mean:.6f}±{e_std:.6f}" if e_std is not None else f"{e_mean:.6f}"
        line = f"{method:<12}{clean_str:>22}{attack_str:>24}{e_str:>24}"
        print(line)
        logging.info(line)
        table_rows.append(
            {
                "Method": method,
                "Clean Mean": clean_mean,
                "Clean Std": clean_std,
                "Attack Mean": attack_mean,
                "Attack Std": attack_std,
                "E Norm Mean": e_mean,
                "E Norm Std": e_std,
            }
        )
    if table_rows:
        try:
            import pandas as pd
            df = pd.DataFrame(table_rows)
            excel_path = os.path.join(output_folder, "multi_attack_summary.xlsx")
            df.to_excel(excel_path, index=False)
            print(f"Saved aggregated summary to {excel_path}")
            logging.info(f"Saved aggregated summary to {excel_path}")
        except Exception as exc:
            logging.warning(f"Failed to save Excel summary ({exc}). Falling back to CSV.")
            print(f"Failed to save Excel summary ({exc}). Falling back to CSV.")
            csv_path = os.path.join(output_folder, "multi_attack_summary.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=[
                        "Method",
                        "Clean Mean",
                        "Clean Std",
                        "Attack Mean",
                        "Attack Std",
                        "E Norm Mean",
                        "E Norm Std",
                    ],
                )
                writer.writeheader()
                for row in table_rows:
                    writer.writerow(row)
            print(f"Saved aggregated summary to {csv_path}")
            logging.info(f"Saved aggregated summary to {csv_path}")


def print_linf_attack_global_summary(output_folder: str):
    if not linf_attack_global_stats:
        return
    title = "Aggregated AdaTR L∞ Experiment Summary"
    header = f"{'Method':<18}{'Clean Mean±Std':>20}{'F Attack Mean±Std':>24}{'L∞ Attack Mean±Std':>26}{'||E||F Mean±Std':>22}{'||E||∞ Mean±Std':>24}"
    print(title)
    print(header)
    print('-' * len(header))
    logging.info(title)
    logging.info(header)
    logging.info('-' * len(header))
    table_rows = []
    for method, values in linf_attack_global_stats.items():
        clean_vals = values.get("clean", [])
        f_vals = values.get("f_attack", [])
        linf_vals = values.get("linf_attack", [])
        fE_vals = values.get("f_E", [])
        e_vals = values.get("linf_E", [])
        clean_mean = np.mean(clean_vals) if clean_vals else None
        clean_std = np.std(clean_vals) if clean_vals else None
        f_mean = np.mean(f_vals) if f_vals else None
        f_std = np.std(f_vals) if f_vals else None
        linf_mean = np.mean(linf_vals) if linf_vals else None
        linf_std = np.std(linf_vals) if linf_vals else None
        fE_mean = np.mean(fE_vals) if fE_vals else None
        fE_std = np.std(fE_vals) if fE_vals else None
        e_mean = np.mean(e_vals) if e_vals else None
        e_std = np.std(e_vals) if e_vals else None

        def format_mean_std(mean_val, std_val):
            if mean_val is None:
                return "--"
            if std_val is None:
                return f"{mean_val:.6f}"
            return f"{mean_val:.6f}±{std_val:.6f}"

        clean_str = format_mean_std(clean_mean, clean_std)
        f_str = format_mean_std(f_mean, f_std)
        linf_str = format_mean_std(linf_mean, linf_std)
        fE_str = format_mean_std(fE_mean, fE_std)
        e_str = format_mean_std(e_mean, e_std)

        line = f"{method:<18}{clean_str:>20}{f_str:>24}{linf_str:>26}{fE_str:>22}{e_str:>24}"
        print(line)
        logging.info(line)
        table_rows.append(
            {
                "Method": method,
                "Clean Mean": clean_mean,
                "Clean Std": clean_std,
                "F Attack Mean": f_mean,
                "F Attack Std": f_std,
                "Linf Attack Mean": linf_mean,
                "Linf Attack Std": linf_std,
                "E F Mean": fE_mean,
                "E F Std": fE_std,
                "E Inf Mean": e_mean,
                "E Inf Std": e_std,
            }
        )
    if table_rows:
        try:
            import pandas as pd
            df = pd.DataFrame(table_rows)
            excel_path = os.path.join(output_folder, "linf_attack_summary.xlsx")
            df.to_excel(excel_path, index=False)
            print(f"Saved L∞ summary to {excel_path}")
            logging.info(f"Saved L∞ summary to {excel_path}")
        except Exception as exc:
            logging.warning(f"Failed to save L∞ Excel summary ({exc}). Falling back to CSV.")
            print(f"Failed to save L∞ Excel summary ({exc}). Falling back to CSV.")
            csv_path = os.path.join(output_folder, "linf_attack_summary.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=[
                        "Method",
                        "Clean Mean",
                        "Clean Std",
                        "F Attack Mean",
                        "F Attack Std",
                        "Linf Attack Mean",
                        "Linf Attack Std",
                        "E F Mean",
                        "E F Std",
                        "E Inf Mean",
                        "E Inf Std",
                    ],
                )
                writer.writeheader()
                for row in table_rows:
                    writer.writerow(row)
            print(f"Saved L∞ summary to {csv_path}")
            logging.info(f"Saved L∞ summary to {csv_path}")


def run_multi_decomposition_attack_suite(
    X: torch.Tensor,
    config: OptimizationConfig,
    base_name: str,
    tr_cores_init: List[torch.Tensor],
):
    if not multi_decomposition_attack_enable:
        return

    method_specs = []
    if attack_multi_tr_enable:
        method_specs.append(
            {
                "label": "TR",
                "method": "TR",
                "tr_core_init": tr_cores_init,
            }
        )
    if attack_multi_tt_enable:
        method_specs.append(
            {
                "label": "TT",
                "method": "TT",
                "tt_rank": tt_attack_rank,
            }
        )
    if attack_multi_tucker_enable:
        method_specs.append(
            {
                "label": "Tucker",
                "method": "Tucker",
                "tucker_rank": tucker_attack_rank,
            }
        )
    if attack_multi_cp_enable:
        method_specs.append(
            {
                "label": "CP",
                "method": "CP",
                "cp_rank": cp_attack_rank,
            }
        )

    if not method_specs:
        print("Multi-decomposition attack enabled but no methods selected.")
        return

    results_rows: List[Dict[str, Optional[float]]] = []
    for spec in method_specs:
        label = spec["label"]
        method_name = spec["method"]
        tr_init = spec.get("tr_core_init")
        tt_rank_val = spec.get("tt_rank")
        tucker_rank_val = spec.get("tucker_rank")
        cp_rank_val = spec.get("cp_rank")
        inner_iters = max(1, multi_attack_inner_iters)
        try:
            recon_clean, _ = reconstruct_tensor_with_method(
                X,
                method_name,
                config,
                tr_core_init=tr_init,
                tt_rank=tt_rank_val,
                tucker_rank=tucker_rank_val if tucker_rank_val is not None else None,
                cp_rank=cp_rank_val,
                inner_num_steps=inner_iters,
            )
            clean_error = torch.norm(X - recon_clean).item()
        except Exception as exc:
            logging.exception(f"[{label}] Clean reconstruction failed: {exc}")
            print(f"[{label}] Clean reconstruction failed: {exc}")
            results_rows.append({"method": label, "clean_error": None, "attack_error": None})
            continue

        attack_error = None
        attack_e_norm = None
        try:
            with peak_memory_monitor(f"{label}_Attack"):
                E_attack, recon_attack, _, attack_e_norm = optimize_als_attack(
                    X,
                    config,
                    method_name,
                    tr_core_init=tr_init,
                    tt_rank=tt_rank_val,
                    tucker_rank=tucker_rank_val if tucker_rank_val is not None else None,
                    cp_rank=cp_rank_val,
                    inner_num_steps=inner_iters,
                    label=label,
                )
            attack_error = torch.norm(X - recon_attack).item()
        except Exception as exc:
            logging.exception(f"[{label}] Attack optimization failed: {exc}")
            print(f"[{label}] Attack optimization failed: {exc}")

        results_rows.append(
            {
                "method": label,
                "clean_error": clean_error,
                "attack_error": attack_error,
                "e_norm": attack_e_norm,
            }
        )

    print_multi_attack_results(base_name, results_rows)
    record_multi_attack_results(base_name, results_rows)
    return results_rows


def plot_attack_distribution_figures(
    attack_label,
    base_name,
    output_dir,
    input_array,
    input_fft_array,
    noise_array,
    attacked_array,
    reconstruction_array,
):
    if input_array is None or noise_array is None or attacked_array is None or reconstruction_array is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    safe_label = attack_label.replace(' ', '_')
    if input_fft_array is None:
        input_fft_array = fft_magnitude(input_array)

    spatial_series = [
        ("Input", input_array),
        ("Noise", noise_array),
        ("Input + Noise", attacked_array),
        ("Reconstruction", reconstruction_array),
    ]
    spatial_title = f"{base_name} - {attack_label} (Spatial)"
    spatial_path = os.path.join(output_dir, f"{base_name}_{safe_label}_spatial.png")
    _plot_distribution_grid(spatial_series, spatial_title, spatial_path)
    spatial_sv_series = [
        (f"{label} SV", compute_channel_singular_values(data)) for label, data in spatial_series
    ]
    spatial_sv_path = os.path.join(output_dir, f"{base_name}_{safe_label}_spatial_svd.png")
    _plot_singular_value_grid(spatial_sv_series, f"{base_name} - {attack_label} (Spatial SV)", spatial_sv_path)

    noise_fft = fft_magnitude(noise_array)
    attacked_fft = fft_magnitude(attacked_array)
    recon_fft = fft_magnitude(reconstruction_array)
    freq_series = [
        ("Input |FFT|", input_fft_array),
        ("Noise |FFT|", noise_fft),
        ("Input+Noise |FFT|", attacked_fft),
        ("Reconstruction |FFT|", recon_fft),
    ]
    freq_title = f"{base_name} - {attack_label} (Fourier)"
    freq_path = os.path.join(output_dir, f"{base_name}_{safe_label}_fft.png")
    _plot_distribution_grid(freq_series, freq_title, freq_path)
    freq_sv_series = [
        (f"{label} SV", compute_channel_singular_values(data)) for label, data in freq_series
    ]
    freq_sv_path = os.path.join(output_dir, f"{base_name}_{safe_label}_fft_svd.png")
    _plot_singular_value_grid(freq_sv_series, f"{base_name} - {attack_label} (Fourier SV)", freq_sv_path)


def generate_low_rank_tensor(shape=(50, 50, 50), ranks=(5, 5, 5), device="cuda", normalize=True, seed=None):
    """
    Generate a synthetic tensor with a prescribed tensor ring rank.

    Args:
        shape: Desired tensor shape.
        ranks: Tensor ring ranks for each mode (cyclic).
        device: Target device for the generated tensor.
        normalize: Whether to scale the tensor entries to [0, 1].
        seed: Optional random seed for reproducibility.

    Returns:
        torch.Tensor: Tensor ring low-rank tensor on the requested device.
    """
    if len(shape) != len(ranks):
        raise ValueError("shape and ranks must have the same length")

    if seed is not None:
        torch.manual_seed(seed)

    shape = list(shape)
    ranks = list(ranks)
    ranks_extended = ranks + [ranks[0]]
    cores = tr.initialize_tr_cores(shape, ranks_extended, device=device)
    tensor = tl.tr_to_tensor(cores)

    tensor = tensor.to(device=device, dtype=torch.float32)

    if normalize:
        tensor = tensor - tensor.min()
        max_val = tensor.max()
        if max_val > 0:
            tensor = tensor / max_val

    return tensor


def objective_function_theta_defence(
    X_clean: torch.Tensor,
    X_noisy: torch.Tensor,
    G_list: List[torch.Tensor],
    Theta: torch.Tensor,
    inner_num_steps: int = 100,
    inner_tol: float = 1e-10,
    inner_dis_num: int = 2000,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    Defence objective: minimize ||X_clean - TR(G(Theta))||_F^2 where
    G(Theta) solves min_G ||X_noisy - Theta - TR(G)||_F^2.
    """
    corrected = X_noisy - Theta
    tr_cores = tr.tr_decompose2(
        corrected,
        G_list,
        max_iter=inner_num_steps,
        tol=inner_tol,
        dis_num=inner_dis_num,
        device=corrected.device,
    )
    TN_G = tl.tr_to_tensor(tr_cores)
    loss = torch.norm(X_clean - TN_G) ** 2
    return loss, TN_G, tr_cores

def objective_function_DFTR(X, G_list, E, inner_num_steps = 100, inner_tol = 1e-10):
    """
    Compute the objective function ||X - TN([G^(k)])||_F^2.
    X: Original tensor (PyTorch tensor)
    G_list: List of factor tensors in tensor train format
    E: Perturbation tensor (PyTorch tensor)
    """

    X_tube = X + E

    # 克隆一份核心列表，避免在 ALS 中原地修改外部传入的 G_list，
    # 这样每次前向都会构建全新的计算图，防止跨迭代复用导致的 backward 异常。
    cloned_cores = [core.clone() for core in G_list]

    tr_cores = tr.tr_decompose4(
        X_tube,
        cloned_cores,
        max_iter=inner_num_steps,
        tol=inner_tol,
        dis_num=2000,
        device="cuda",
        track_grad = True,
    )

    TN_G = tl.tr_to_tensor(tr_cores)  # Reconstruct tensor from tensor train

    return -torch.norm(X - TN_G) ** 2, TN_G, tr_cores

def objective_function_DFTR_noisy(
    X: torch.Tensor,
    G_list: List[torch.Tensor],
    E: torch.Tensor,
    inner_num_steps: int = 100,
    inner_tol: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    Variant of DF-ADdaTR objective that measures the reconstruction error against
    the attacked observation (X + E).
    """
    X_noisy = X + E

    cloned_cores = [core.clone() for core in G_list]

    tr_cores = tr.tr_decompose4(
        X_noisy,
        cloned_cores,
        max_iter=inner_num_steps,
        tol=inner_tol,
        dis_num=2000,
        device="cuda",
        track_grad=True,
    )

    TN_G = tl.tr_to_tensor(tr_cores)
    return -torch.norm(X_noisy - TN_G) ** 2, TN_G, tr_cores

def objective_function_ATR(X, G_list, E, inner_num_steps, inner_tol):
    """
    Compute the comparison objective function.
    X: Original tensor (PyTorch tensor)
    G_list: List of factor tensors in tensor train format
    E: Perturbation tensor (PyTorch tensor)
    inner_num_steps: Number of inner optimization steps
    inner_tol: Tolerance for inner optimization
    """

    # Example: Use a different decomposition method or a different loss function
    EC = E.clone().detach()
    tr_cores = tr.tr_decompose2(X+EC, G_list, max_iter=inner_num_steps, tol=inner_tol, dis_num = 2000, device="cuda")
    TN_G = tl.tr_to_tensor(tr_cores)  # Reconstruct tensor from tensor train

    loss_values = []

    return -torch.norm(X+E - TN_G) ** 2, TN_G, tr_cores  # Example: Minimize the norm instead of maximizing


def optimize_theta_defence(
    clean_tensor: torch.Tensor,
    attacked_tensor: torch.Tensor,
    G_list_init: List[torch.Tensor],
    config: OptimizationConfig,
    initial_theta: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """
    Defence counterpart of optimize_perturbation.
    Learns a bounded correction tensor Theta that minimizes
    ||clean_tensor - TR(G^{(T)}(Theta))||_F^2 with
    G^{(T)}(Theta) = argmin_G ||attacked_tensor - Theta - TR(G)||_F^2.
    """
    theta_curve: List[float] = []
    theta_eps = config.theta_epsilon
    if theta_eps is None:
        theta_eps = config.attr_epsilon if config.attr_epsilon is not None else config.epsilon

    if initial_theta is not None:
        Theta = initial_theta.clone().detach().to(attacked_tensor.device)
        Theta.requires_grad_(True)
    else:
        Theta = torch.zeros_like(attacked_tensor, requires_grad=True, device=attacked_tensor.device)

    if theta_eps is not None:
        norm_sq = torch.norm(Theta).pow(2)
        if norm_sq > theta_eps and norm_sq > 0:
            Theta.data.mul_((theta_eps / (norm_sq + 1e-12)) ** 0.5)

    defence_lr = getattr(config, "defence_lr", getattr(config, "lr", 0.01))
    optimizer = torch.optim.Adam([Theta], lr=defence_lr)

    working_cores = [core.clone().detach() for core in G_list_init]
    TN_G_detached: Optional[torch.Tensor] = None

    for step in range(config.outer_num_steps_defence):
        optimizer.zero_grad()

        Theta_old = Theta.detach().clone()

        loss, TN_G, new_cores = objective_function_theta_defence(
            clean_tensor,
            attacked_tensor,
            working_cores,
            Theta,
            inner_num_steps=config.inner_num_steps,
            inner_tol=config.inner_tol,
            inner_dis_num=config.inner_dis_num,
        )

        if torch.isnan(loss):
            raise ValueError("Theta defence loss became NaN")

        loss.backward()
        optimizer.step()

        working_cores = [core.detach() for core in new_cores]
        TN_G_detached = TN_G.detach()

        if theta_eps is not None:
            norm_sq = torch.norm(Theta).pow(2)
            if norm_sq > theta_eps and norm_sq > 0:
                Theta.data.mul_((theta_eps / (norm_sq + 1e-12)) ** 0.5)

        if torch.isnan(Theta).any():
            raise ValueError("Theta contains NaN after update")

        recon_err = torch.norm(clean_tensor - TN_G_detached).item()
        theta_curve.append(recon_err)

        theta_delta = torch.norm(Theta.detach() - Theta_old)
        theta_prev_norm = torch.norm(Theta_old)
        rel_change = theta_delta / (theta_prev_norm + 1e-12)
        rel_change_val = rel_change.item()
        theta_norm_val = torch.norm(Theta).item()
        logging.info(
            f"[Theta-Defence] Step {step + 1}/{config.outer_num_steps_defence}, "
            f"loss={loss.item():.6f}, recon_norm={recon_err:.6f}, ||Theta||={theta_norm_val:.6f}, "
            f"rel_change={rel_change_val:.6f}"
        )
        print(
            f"[Theta-Defence] Step {step + 1}/{config.outer_num_steps_defence}: "
            f"loss={loss.item():.6f}, recon_norm={recon_err:.6f}, ||Theta||={theta_norm_val:.6f}, "
            f"rel_change={rel_change_val:.6f}"
        )

        if rel_change_val < config.outer_tol:
            logging.info(
                f"[Theta-Defence] Early stop at step {step + 1} because relative Theta change={rel_change_val:.3e} < {config.outer_tol}"
            )
            print(
                f"[Theta-Defence] Early stop at step {step + 1} due to relative Theta change {rel_change_val:.3e} < {config.outer_tol}"
            )
            break

    if TN_G_detached is None:
        TN_G_detached = tl.tr_to_tensor(working_cores).detach()
    return Theta.detach(), TN_G_detached, theta_curve

def optimize_G_list(X, G_list_ori, E, config: OptimizationConfig):
    """
    Optimize the tensor train cores G_list_ori using gradient descent.
    X: Original tensor (PyTorch tensor)
    G_list_ori: List of factor tensors in tensor train format (PyTorch tensors)
    config: Optimization configuration
    """
    # # 确保 G_list_ori 的每个张量启用梯度
    # G_list = [G.clone().detach().requires_grad_(True) for G in G_list_ori]
    G_list = []
    trainable_cores = []
    for idx, G in enumerate(G_list_ori):
        core = G.clone().detach()
        if idx != 0:  # 0 号核心不做防御优化
            core.requires_grad_(True)
            trainable_cores.append(core)
        G_list.append(core)

    # 定义优化器，只优化 G_list
    # optimizer = torch.optim.Adam(G_list, lr=config.lr)
    defence_lr = getattr(config, "defence_lr", getattr(config, "lr", 0.01))
    optimizer = torch.optim.Adam(trainable_cores, lr=defence_lr)
    # optimizer = torch.optim.SGD(trainable_cores, lr=defence_lr, momentum=0.9)


    loss_list = []

    for step in range(config.outer_num_steps_defence):
        prev_trainable = [core.detach().clone() for core in trainable_cores]
        optimizer.zero_grad()  # 清除梯度

        # 计算损失（负数）
        loss, TN_G, _ = objective_function_DFTR(X, G_list, E, inner_num_steps=config.inner_num_steps, inner_tol=config.inner_tol)
        loss = -loss  # 最小化负数的损失

        # for idx, core in enumerate(G_list):
        #     core.register_hook(make_grad_printer(f"core{idx}"))

        # 反向传播
        loss.backward()
        
        TN_G_det = TN_G.detach()
        G_list_det = [core.detach() for core in G_list]

        # 更新 G_list
        grad_norms = []
        for core in G_list:
            if core.grad is not None:
                grad_norms.append(core.grad.norm().item())
            else:
                grad_norms.append(float('nan'))

        grad_msg = ", ".join(
            f"core{idx}: {norm:.6f}" if not math.isnan(norm) else f"core{idx}: nan"
            for idx, norm in enumerate(grad_norms)
        )
        print(f"[DF-ADdaTR] Grad norms step {step + 1}: {grad_msg}")
        logging.info(f"[DF-ADdaTR] Grad norms step {step + 1}: {grad_msg}")

        # torch.nn.utils.clip_grad_norm_(trainable_cores, max_norm=150.0)
        optimizer.step()

        # 记录损失
        loss_value = loss.item()
        loss_list.append(loss_value)

        recon_norm = torch.norm(X - TN_G).item()
        noise_norm = torch.norm(E).item()

        log_msg = (
            f"[DF-ADdaTR] Step {step + 1}/{config.outer_num_steps_defence}: "
            f"loss={loss_value:.6f}, recon_norm={recon_norm:.6f}, ||E||={noise_norm:.6f}"
        )
        print(log_msg)
        logging.info(log_msg)

        # 检查是否有 NaN
        if torch.isnan(loss):
            raise ValueError("Loss is NaN")
        rel_change = 0.0
        for prev_core, core in zip(prev_trainable, trainable_cores):
            delta = torch.norm(core.detach() - prev_core)
            base = torch.norm(prev_core)
            rel = delta / (base + 1e-12) if base > 0 else delta
            rel_change = max(rel_change, rel)
        if rel_change < config.outer_tol:
            print(f"[DF-ADdaTR] Early stop at step {step + 1} due to relative core change {rel_change:.3e} < {config.outer_tol}")
            logging.info(f"[DF-ADdaTR] Early stop at step {step + 1} because relative core change={rel_change:.3e} < {config.outer_tol}")
            break

    # 返回优化后的 G_list 和损失列表
    return G_list, loss_list


def optimize_G_list_noisy(X, G_list_ori, E, config: OptimizationConfig):
    """
    Optimise tensor-ring cores against the attacked observation X + E,
    using the DF-ADdaTR objective that matches the noisy tensor.
    """
    G_list: List[torch.Tensor] = []
    trainable_cores: List[torch.Tensor] = []
    for idx, G in enumerate(G_list_ori):
        core = G.clone().detach()
        if idx != 0:
            core.requires_grad_(True)
            trainable_cores.append(core)
        G_list.append(core)

    defence_lr = getattr(config, "defence_lr", getattr(config, "lr", 0.01))
    optimizer = torch.optim.Adam(trainable_cores, lr=defence_lr)

    loss_list: List[float] = []
    attacked_tensor = X + E

    for step in range(config.outer_num_steps_defence):
        prev_trainable = [core.detach().clone() for core in trainable_cores]
        optimizer.zero_grad()

        loss, TN_G, _ = objective_function_DFTR_noisy(
            X,
            G_list,
            E,
            inner_num_steps=config.inner_num_steps,
            inner_tol=config.inner_tol,
        )
        loss = -loss

        loss.backward()

        grad_norms = []
        for core in G_list:
            if core.grad is not None:
                grad_norms.append(core.grad.norm().item())
            else:
                grad_norms.append(float("nan"))

        grad_msg = ", ".join(
            f"core{idx}: {norm:.6f}" if not math.isnan(norm) else f"core{idx}: nan"
            for idx, norm in enumerate(grad_norms)
        )
        print(f"[DF-ADdaTR-Noisy] Grad norms step {step + 1}: {grad_msg}")
        logging.info(f"[DF-ADdaTR-Noisy] Grad norms step {step + 1}: {grad_msg}")

        optimizer.step()

        loss_value = loss.item()
        loss_list.append(loss_value)

        recon_norm = torch.norm(X - TN_G).item()
        noise_norm = torch.norm(E).item()

        log_msg = (
            f"[DF-ADdaTR-Noisy] Step {step + 1}/{config.outer_num_steps_defence}: "
            f"loss={loss_value:.6f}, recon_norm={recon_norm:.6f}, ||E||={noise_norm:.6f}"
        )
        print(log_msg)
        logging.info(log_msg)

        if torch.isnan(loss):
            raise ValueError("Loss is NaN")

        rel_change = 0.0
        for prev_core, core in zip(prev_trainable, trainable_cores):
            delta = torch.norm(core.detach() - prev_core)
            base = torch.norm(prev_core)
            rel = delta / (base + 1e-12) if base > 0 else delta
            rel_change = max(rel_change, rel)
        if rel_change < config.outer_tol:
            msg = (
                f"[DF-ADdaTR-Noisy] Early stop at step {step + 1} "
                f"due to relative core change {rel_change:.3e} < {config.outer_tol}"
            )
            print(msg)
            logging.info(msg)
            break

    return G_list, loss_list


def _generate_lr_values(min_lr: float, max_lr: float, num_points: int) -> List[float]:
    if num_points <= 0:
        return []
    if min_lr <= 0 or max_lr <= 0:
        return []
    if min_lr == max_lr:
        return [float(min_lr)]
    lr_array = np.logspace(np.log10(min_lr), np.log10(max_lr), num_points)
    return [float(lr) for lr in lr_array]


def _save_lr_finder_results(lrs: List[float], losses: List[float], output_dir: str, filename_prefix: str, title: str, ylabel: str = 'Loss'):
    if not lrs:
        return
    os.makedirs(output_dir, exist_ok=True)
    safe_prefix = filename_prefix.replace(' ', '_')
    data = np.column_stack([np.asarray(lrs, dtype=np.float64), np.asarray(losses, dtype=np.float64)])
    csv_path = os.path.join(output_dir, f"{safe_prefix}.csv")
    np.savetxt(csv_path, data, delimiter=',', header='lr,loss', comments='')

    plt.figure(figsize=(8, 5))
    plt.plot(lrs, losses, marker='o')
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    png_path = os.path.join(output_dir, f"{safe_prefix}.png")
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close()


def _save_step_curve(losses: List[float], output_dir: str, filename_prefix: str, title: str, ylabel: str = 'Loss / Error'):
    if not losses:
        return
    os.makedirs(output_dir, exist_ok=True)
    safe_prefix = filename_prefix.replace(' ', '_')
    steps = np.arange(1, len(losses) + 1, dtype=np.int64)
    data = np.column_stack([steps, np.asarray(losses, dtype=np.float64)])
    csv_path = os.path.join(output_dir, f"{safe_prefix}_curve.csv")
    np.savetxt(csv_path, data, delimiter=',', header='step,loss', comments='')

    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, marker='o')
    plt.xlabel('Step')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    png_path = os.path.join(output_dir, f"{safe_prefix}_curve.png")
    plt.savefig(png_path, dpi=200, bbox_inches='tight')
    plt.close()


def _run_attr_lr_finder(X: torch.Tensor, perturbation: torch.Tensor, tr_cores_init: List[torch.Tensor], base_cfg: OptimizationConfig, lr_values: List[float], base_name: str, label: str, output_dir: str):
    if not lr_values:
        return
    attacked_tensor = (X + perturbation).detach()
    results: List[float] = []
    for idx, lr in enumerate(lr_values):
        cfg_lr = clone_config(base_cfg)
        cfg_lr.defence_lr = float(lr)
        losses_curve: List[float] = []
        try:
            _, TN_G_attr, attr_losses = optimize_perturbation_comparison(
                attacked_tensor,
                clone_tr_cores(tr_cores_init),
                cfg_lr,
                attr_epsilon=cfg_lr.attr_epsilon,
                clean_tensor=X,
            )
            losses_curve = [float(val) for val in attr_losses]
            loss_val = torch.norm(X - TN_G_attr).item()
            results.append(loss_val)
        except Exception as exc:
            logging.warning(f"LR finder (ATTR-{label}) failed at lr={lr}: {exc}")
            results.append(float('nan'))
        curve_prefix = f"{base_name}_{label.replace(' ', '_')}_ATTR_lr_{idx:02d}"
        _save_step_curve(
            losses_curve,
            output_dir,
            curve_prefix,
            f"ATTR Loss Curve ({base_name}-{label}, lr={lr:.2e})",
            ylabel='Reconstruction Error',
        )
    filename = f"{base_name}_{label.replace(' ', '_')}_ATTR_lr_finder"
    _save_lr_finder_results(lr_values, results, output_dir, filename, f"ATTR LR Finder ({base_name}-{label})", ylabel='Reconstruction Error')


def _run_df_lr_finder(X: torch.Tensor, perturbation: torch.Tensor, tr_cores_init: List[torch.Tensor], base_cfg: OptimizationConfig, lr_values: List[float], base_name: str, label: str, output_dir: str):
    if not lr_values:
        return
    results: List[float] = []
    for idx, lr in enumerate(lr_values):
        cfg_lr = clone_config(base_cfg)
        cfg_lr.defence_lr = float(lr)
        losses_curve: List[float] = []
        try:
            optimized_cores, loss_list = optimize_G_list(X, clone_tr_cores(tr_cores_init), perturbation, cfg_lr)
            losses_curve = [float(val) for val in loss_list]
            with torch.no_grad():
                _, TN_G_eval, _ = objective_function_DFTR(
                    X,
                    [core.clone().detach() for core in optimized_cores],
                    perturbation,
                    inner_num_steps=cfg_lr.inner_num_steps,
                    inner_tol=cfg_lr.inner_tol,
                )
                loss_val = torch.norm(X - TN_G_eval).item()
            results.append(loss_val)
        except Exception as exc:
            logging.warning(f"LR finder (DF-ADdaTR-{label}) failed at lr={lr}: {exc}")
            results.append(float('nan'))
        curve_prefix = f"{base_name}_{label.replace(' ', '_')}_DF_ADdaTR_lr_{idx:02d}"
        _save_step_curve(
            losses_curve,
            output_dir,
            curve_prefix,
            f"DF-ADdaTR Loss Curve ({base_name}-{label}, lr={lr:.2e})",
            ylabel='Reconstruction Error',
        )
    filename = f"{base_name}_{label.replace(' ', '_')}_DF_ADdaTR_lr_finder"
    _save_lr_finder_results(lr_values, results, output_dir, filename, f"DF-ADdaTR LR Finder ({base_name}-{label})", ylabel='Reconstruction Error')


def _run_attr_df_lr_finder(X: torch.Tensor, perturbation: torch.Tensor, tr_cores_init: List[torch.Tensor], base_cfg: OptimizationConfig, lr_values: List[float], base_name: str, label: str, output_dir: str):
    if not lr_values:
        return
    results: List[float] = []
    attacked_tensor = (X + perturbation).detach()
    for idx, lr in enumerate(lr_values):
        cfg_lr = clone_config(base_cfg)
        cfg_lr.defence_lr = float(lr)
        losses_curve: List[float] = []
        try:
            optimized_cores, _ = optimize_G_list(X, clone_tr_cores(tr_cores_init), perturbation, cfg_lr)
            _, TN_G_attr_df, attr_losses_df = optimize_perturbation_comparison(
                attacked_tensor,
                [core.clone().detach() for core in optimized_cores],
                cfg_lr,
                attr_epsilon=cfg_lr.attr_epsilon,
                clean_tensor=X,
            )
            losses_curve = [float(val) for val in attr_losses_df]
            loss_val = torch.norm(X - TN_G_attr_df).item()
            results.append(loss_val)
        except Exception as exc:
            logging.warning(f"LR finder (ATTR(DF)-{label}) failed at lr={lr}: {exc}")
            results.append(float('nan'))
        curve_prefix = f"{base_name}_{label.replace(' ', '_')}_ATTR_DF_lr_{idx:02d}"
        _save_step_curve(
            losses_curve,
            output_dir,
            curve_prefix,
            f"ATTR(DF) Loss Curve ({base_name}-{label}, lr={lr:.2e})",
            ylabel='Reconstruction Error',
        )
    filename = f"{base_name}_{label.replace(' ', '_')}_ATTR_DF_lr_finder"
    _save_lr_finder_results(lr_values, results, output_dir, filename, f"ATTR(DF) LR Finder ({base_name}-{label})", ylabel='Reconstruction Error')


def _run_ata_lr_finder(
    X: torch.Tensor,
    perturbation: torch.Tensor,
    tr_cores_init: List[torch.Tensor],
    attack_cfg_template: OptimizationConfig,
    defence_cfg_template: OptimizationConfig,
    lr_values: List[float],
    base_name: str,
    label: str,
    output_dir: str,
):
    if not lr_values:
        return
    results: List[float] = []
    perturbation_type = "learned"
    label_lower = label.lower()
    if label_lower == "clean":
        perturbation_type = "clean"
    elif label_lower == "gaussian":
        perturbation_type = "gaussian"
    elif label_lower == "ours_f":
        perturbation_type = "simplified"
    for idx, lr in enumerate(lr_values):
        defence_cfg_lr = clone_config(defence_cfg_template)
        defence_cfg_lr.defence_lr = float(lr)
        attack_cfg_lr = clone_config(attack_cfg_template)
        losses_curve: List[float] = []
        try:
            ata_result = alternating_attack_training(
                X,
                clone_tr_cores(tr_cores_init),
                attack_cfg_lr,
                defence_cfg_lr,
                num_rounds=defence_cfg_lr.outer_num_steps_defence,
                track_history=True,
                store_perturbations=False,
                base_perturbation=perturbation,
                perturbation_type=perturbation_type,
            )
            history = ata_result.get('history', [])
            for entry in history:
                defence_curve = entry.get('defence_loss_curve', [])
                if defence_curve:
                    losses_curve.extend([float(v) for v in defence_curve])
            final_loss = ata_result.get('best_loss', float('nan'))
            results.append(float(final_loss))
        except Exception as exc:
            logging.warning(f"LR finder (ATAdaTR-{label}) failed at lr={lr}: {exc}")
            results.append(float('nan'))
        curve_prefix = f"{base_name}_{label.replace(' ', '_')}_ATAdaTR_lr_{idx:02d}"
        _save_step_curve(
            losses_curve,
            output_dir,
            curve_prefix,
            f"ATAdaTR Loss Curve ({base_name}-{label}, lr={lr:.2e})",
            ylabel='Reconstruction Error',
        )
    filename = f"{base_name}_{label.replace(' ', '_')}_ATAdaTR_lr_finder"
    _save_lr_finder_results(lr_values, results, output_dir, filename, f"ATAdaTR LR Finder ({base_name}-{label})", ylabel='Final Defence Loss')


def _clone_cores(cores):
    return [core.clone().detach() for core in cores]


def _evaluate_defence(X, cores, perturbation, cfg):
    with torch.no_grad():
        _, TN_G_eval, _ = objective_function_DFTR(
            X,
            _clone_cores(cores),
            perturbation,
            inner_num_steps=cfg.inner_num_steps,
            inner_tol=cfg.inner_tol,
        )
        return torch.norm(X - TN_G_eval).item()


def alternating_attack_training(
    X,
    G_list_init,
    attack_config,
    defence_config,
    num_rounds,
    track_history=False,
    store_perturbations=False,
    base_perturbation=None,
    perturbation_type="learned",
):
    if num_rounds <= 0:
        raise ValueError('num_rounds must be a positive integer')

    original_attack_steps = attack_config.outer_num_steps_attack
    original_defence_steps = defence_config.outer_num_steps_defence
    attack_config.outer_num_steps_attack = 1
    defence_config.outer_num_steps_defence = 1

    current_G = _clone_cores(G_list_init)
    best_state = _clone_cores(current_G)
    best_loss = float('inf')
    history = []
    attack_E = None
    base_norm = None

    if base_perturbation is not None:
        base_perturbation = base_perturbation.detach()
        base_norm = torch.norm(base_perturbation).item()

    try:
        for round_idx in range(num_rounds):
            current_round = round_idx + 1
            print(f"ATAdaTR round {current_round}/{num_rounds}: starting attack optimisation")
            initial_attack_E = attack_E
            if perturbation_type == "clean" or (base_norm is not None and base_norm == 0):
                attack_E = torch.zeros_like(X, device=X.device)
                attack_losses = []
            elif perturbation_type == "gaussian":
                if initial_attack_E is not None:
                    attack_E = initial_attack_E.clone().detach().to(X.device)
                else:
                    attack_E = torch.randn_like(X, device=X.device)
                attack_E_norm = torch.norm(attack_E).item()
                target_norm = base_norm if base_norm is not None and base_norm > 0 else math.sqrt(attack_config.epsilon)
                if attack_E_norm > 0:
                    attack_E = attack_E * (target_norm / (attack_E_norm + 1e-12))
                attack_losses = []
            elif perturbation_type == "simplified":
                attack_E, _, attack_losses = optimize_perturbation_simplifed(
                    X,
                    _clone_cores(current_G),
                    attack_config,
                    initial_E=initial_attack_E,
                )
            else:
                attack_E, _, attack_losses = optimize_perturbation(
                    X,
                    _clone_cores(current_G),
                    attack_config,
                    initial_E=initial_attack_E,
                )

            optimized_G_list, defence_losses = optimize_G_list(
                X,
                _clone_cores(current_G),
                attack_E,
                defence_config,
            )
            print(f"ATAdaTR round {current_round}/{num_rounds}: completed defence optimisation")

            current_G = _clone_cores(optimized_G_list)
            loss_last = float(defence_losses[-1]) if defence_losses else float('inf')

            if loss_last < best_loss:
                best_loss = loss_last
                best_state = _clone_cores(current_G)

            if track_history:
                history_entry = {
                    'round': round_idx + 1,
                    'attack_loss_curve': [float(val) for val in attack_losses] if attack_losses else [],
                    'defence_loss_curve': [float(val) for val in defence_losses],
                    'attack_loss_final': float(attack_losses[-1]) if attack_losses else None,
                    'defence_loss_final': loss_last,
                }
                if store_perturbations and attack_E is not None:
                    history_entry['perturbation'] = attack_E.detach().cpu()
                history.append(history_entry)
    finally:
        attack_config.outer_num_steps_attack = original_attack_steps
        defence_config.outer_num_steps_defence = original_defence_steps

    result = {
        'final_G_list': _clone_cores(current_G),
        'best_G_list': _clone_cores(best_state),
        'best_loss': best_loss,
        'last_perturbation': attack_E.detach().clone() if attack_E is not None else None,
    }

    if track_history:
        result['history'] = history

    return result


def run_adversarial_comparison(
    X,
    G_list_init,
    attack_config,
    defence_config,
    adv_rounds,
    track_history=False,
    store_perturbations=False,
    print_table=True,
):
    clean_perturbation = torch.zeros_like(X, device=X.device)
    clean_norm = _evaluate_defence(X, G_list_init, clean_perturbation, defence_config)

    attack_X = X.clone().detach()
    attack_E, attack_TN, attack_losses = optimize_perturbation(
        attack_X,
        _clone_cores(G_list_init),
        attack_config,
    )
    attack_norm = torch.norm(X - attack_TN).item()

    defence_G, defence_losses = optimize_G_list(
        X,
        _clone_cores(G_list_init),
        attack_E,
        defence_config,
    )
    defence_norm = _evaluate_defence(X, defence_G, attack_E, defence_config)

    adv_training_result = alternating_attack_training(
        X,
        _clone_cores(G_list_init),
        attack_config,
        defence_config,
        adv_rounds,
        track_history=track_history,
        store_perturbations=store_perturbations,
    )

    adv_best_cores = adv_training_result['best_G_list']
    adv_attack_X = X.clone().detach()
    adv_attack_E, adv_attack_TN, adv_attack_losses = optimize_perturbation(
        adv_attack_X,
        _clone_cores(adv_best_cores),
        attack_config,
    )
    adv_attack_norm = torch.norm(X - adv_attack_TN).item()

    adv_defence_G, adv_defence_losses = optimize_G_list(
        X,
        _clone_cores(adv_best_cores),
        adv_attack_E,
        defence_config,
    )
    adv_defence_norm = _evaluate_defence(X, adv_defence_G, adv_attack_E, defence_config)

    results = {
        'clean': {'norm': clean_norm},
        'attack': {
            'norm': attack_norm,
            'loss_final': float(attack_losses[-1]) if attack_losses else None,
            'loss_curve': [float(val) for val in attack_losses],
        },
        'defence': {
            'norm': defence_norm,
            'loss_final': float(defence_losses[-1]) if defence_losses else None,
            'loss_curve': [float(val) for val in defence_losses],
        },
        'adversarial_training': {
            'norm': adv_defence_norm,
            'loss_final': float(adv_defence_losses[-1]) if adv_defence_losses else None,
            'loss_curve': [float(val) for val in adv_defence_losses],
            'attack_norm': adv_attack_norm,
            'attack_loss_final': float(adv_attack_losses[-1]) if adv_attack_losses else None,
            'attack_loss_curve': [float(val) for val in adv_attack_losses],
            'best_loss': adv_training_result['best_loss'],
        },
    }

    if store_perturbations:
        results['attack']['perturbation'] = attack_E.detach().cpu()
        results['adversarial_training']['attack_perturbation'] = adv_attack_E.detach().cpu()
        last_pert = adv_training_result.get('last_perturbation')
        if last_pert is not None:
            results['adversarial_training']['last_perturbation'] = last_pert.detach().cpu()

    if track_history and 'history' in adv_training_result:
        results['history'] = adv_training_result['history']

    if print_table:
        header = f"{'Scenario':<32}{'||X - TN||':>15}{'Loss_final':>15}"
        print(header)
        print('-' * len(header))

        clean_row = ('Clean TR_ALS', clean_norm, '--')
        attack_loss_final = results['attack']['loss_final']
        attack_row = (
            'Attack (optimize_perturbation)',
            attack_norm,
            f"{attack_loss_final:.6f}" if attack_loss_final is not None else '--',
        )
        defence_loss_final = results['defence']['loss_final']
        defence_row = (
            'Defence (optimize_G_list)',
            defence_norm,
            f"{defence_loss_final:.6f}" if defence_loss_final is not None else '--',
        )
        adv_loss_final = results['adversarial_training']['loss_final']
        adv_row = (
            'Adversarial Training Defence',
            adv_defence_norm,
            f"{adv_loss_final:.6f}" if adv_loss_final is not None else '--',
        )

        for label, norm_val, loss_val in (clean_row, attack_row, defence_row, adv_row):
            print(f"{label:<32}{norm_val:>15.6f}{loss_val:>15}")

    return results



def process_tensor_data(X, base_name, output_folder, config):
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    else:
        X = X.to(dtype=torch.float32)

    X = X.to("cuda")
    config = scaled_config_for_tensor(config, X)
    peak_memory_records: Dict[str, Optional[int]] = {}
    set_peak_memory_recorder(peak_memory_records)

    # Decompose X into tensor train format
    # ranks = [3, 3, 3]  # Rank for each factor tensor
    # ranks = [5, 5, 5]  # Rank for each factor tensor
    ranks = [3, 3, 3, 3]  # Rank for each factor tensor
    ranks = ranks + [ranks[0]]  # 最后一个 rank 应该等于第一个 rank
    shape = list(X.shape)
    tr_cores = initialize_tr_cores_svd(X, ranks, device="cuda")
    tr_cores_init = clone_tr_cores(tr_cores)
    mode_rank_products = extract_tr_mode_rank_products(tr_cores_init)
    # 将 tr_cores 转换为 NumPy 数组
    tr_cores_np = [core.cpu().numpy() for core in tr_cores]

    # 创建一个字典来存储所有的核心张量
    tr_cores_dict = {f'tr_core_{i}': tr_cores_np[i] for i in range(len(tr_cores_np))}

    if save_mat_enabled:
        # 保存字典到 .mat 文件
        scipy.io.savemat(os.path.join(output_folder, f'tr_cores_{base_name}.mat'), tr_cores_dict)

    E_optimized_F = None
    TN_G_F = None
    loss_ours_F = []
    time_simplified = None
    norm_attack_ours_F = None
    TN_G_ori = None
    TN_G_Gaus = None
    TN_G = None

    if attack_enable_ours_f:
        with peak_memory_monitor("Ours_F"):
            shared_initial_E = torch.randn_like(X, device=X.device)
            shared_initial_E = shared_initial_E / (torch.norm(shared_initial_E) + 1e-12)
            shared_initial_E = shared_initial_E * (config.epsilon ** 0.5)
            start_time = time.time()
            E_optimized_F, TN_G_F, loss_ours_F = optimize_perturbation_simplifed(
                X, clone_tr_cores(tr_cores_init), config, initial_E=shared_initial_E
            )
            time_simplified = time.time() - start_time
            print(f"optimize_perturbation_simplifed 运行时间: {time_simplified:.4f} 秒")
            with torch.no_grad():
                attacked_tensor_f = X + E_optimized_F
                tr_cores_attack_f = tr.tr_decompose2(
                    attacked_tensor_f,
                    clone_tr_cores(tr_cores_init),
                    max_iter=config.inner_num_steps,
                    tol=inner_tol,
                    device="cuda",
                )
                TN_attack_f = tl.tr_to_tensor(tr_cores_attack_f)
                norm_attack_ours_F = torch.norm(X - TN_attack_f).item()
    else:
        print("optimize_perturbation_simplifed 已跳过 (attack_enable_ours_f=False)")

    E_optimized = None
    TN_G = None
    loss_ours = []
    time_optimized = None
    norm_attack_ours = None
    E_attr_attack = None
    TN_G_attr_attack = None
    loss_attr_attack = []
    norm_attack_attr = None

    if attack_enable_ours:
        if 'shared_initial_E' not in locals():
            shared_initial_E = torch.randn_like(X, device=X.device)
            shared_initial_E = shared_initial_E / (torch.norm(shared_initial_E) + 1e-12)
            shared_initial_E = shared_initial_E * (config.epsilon ** 0.5)
        X.requires_grad_(True)
        with peak_memory_monitor("Ours"):
            start_time = time.time()
            E_optimized, TN_G, loss_ours = optimize_perturbation(
                X, clone_tr_cores(tr_cores_init), config, initial_E=shared_initial_E
            )
            time_optimized = time.time() - start_time
            print(f"optimize_perturbation 运行时间: {time_optimized:.4f} 秒")
            with torch.no_grad():
                attacked_tensor = X + E_optimized
                tr_cores_attack = tr.tr_decompose2(
                    attacked_tensor,
                    clone_tr_cores(tr_cores_init),
                    max_iter=config.inner_num_steps,
                    tol=inner_tol,
                    device="cuda",
                )
                TN_attack = tl.tr_to_tensor(tr_cores_attack)
                norm_attack_ours = torch.norm(X - TN_attack).item()
    else:
        print("optimize_perturbation 已跳过 (attack_enable_ours=False)")

    if attack_enable_attr_attack:
        attr_attack_cfg = clone_config(config)
        attr_attack_cfg.attr_epsilon = config.epsilon
        attr_attack_cfg.outer_num_steps_defence = config.outer_num_steps_attack
        attr_attack_cfg.defence_lr = getattr(config, "attack_lr", attr_attack_cfg.defence_lr)
        with peak_memory_monitor("ATTR_Attack"):
            start_time = time.time()
            E_attr_attack, TN_G_attr_attack, loss_attr_attack = optimize_perturbation_comparison(
                X.detach(),
                clone_tr_cores(tr_cores_init),
                attr_attack_cfg,
                attr_epsilon=attr_attack_cfg.attr_epsilon,
                clean_tensor=None,
            )
            print(f"ATTR attack optimize_perturbation_comparison 运行时间: {time.time() - start_time:.4f} 秒")
            with torch.no_grad():
                norm_attack_attr = torch.norm(X - TN_G_attr_attack).item()
            print(f"Norm (ATTR Attack): {norm_attack_attr}")
            logging.info(f"Norm (ATTR Attack): {norm_attack_attr}")
    else:
        print("ATTR attack 已跳过 (attack_enable_attr_attack=False)")

    X = X.detach()
    tr_cores = [G.detach() for G in tr_cores]

    lr_finder_values: List[float] = []
    lr_finder_output_dir: Optional[str] = None
    if lr_finder_enabled:
        lr_finder_values = _generate_lr_values(lr_finder_min_lr, lr_finder_max_lr, lr_finder_num_points)
        if lr_finder_values:
            lr_finder_output_dir = os.path.join(output_folder, lr_finder_folder_name, base_name)
            os.makedirs(lr_finder_output_dir, exist_ok=True)
        else:
            logging.warning("LR finder enabled but no valid learning rate values were generated.")

    norm2_original_TR_algorithm = None
    zero_perturbation = None
    if attack_enable_clean:
        with peak_memory_monitor("Clean"):
            tr_cores2 = tr.tr_decompose2(X, clone_tr_cores(tr_cores_init), max_iter=config.inner_num_steps, tol=inner_tol, device="cuda")
            TN_G_ori = tl.tr_to_tensor(tr_cores2)
            TN_G_ori_F = (TN_G_ori).cpu().detach().numpy()
            save_mat_file(output_folder, f'X_Rec', TN_G_ori_F)
            norm2_original_TR_algorithm = torch.norm(X - TN_G_ori).item()
            zero_perturbation = torch.zeros_like(X, device="cuda")
            print(f"Norm2 (Original TR Algorithm): {norm2_original_TR_algorithm}")
            logging.info(f"Norm2 (Original TR Algorithm): {norm2_original_TR_algorithm}")

    E_Gaus = None
    norm2_Gaus = None
    if attack_enable_gaussian:
        with peak_memory_monitor("Gaussian"):
            E_Gaus = torch.normal(mean=0, std=1, size=X.shape, device="cuda")
            scaling_factor = (config.epsilon ** 0.5) / torch.norm(E_Gaus)
            if torch.norm(E_Gaus)**2 > config.epsilon:
                E_Gaus.data = scaling_factor * E_Gaus
            tr_cores3 = tr.tr_decompose2(X + E_Gaus, clone_tr_cores(tr_cores_init), max_iter=config.inner_num_steps, tol=inner_tol, device="cuda")
            TN_G_Gaus = tl.tr_to_tensor(tr_cores3)
            norm2_Gaus = torch.norm(X - TN_G_Gaus).item()
            print(f"Gaussian noise added with norm: {torch.norm(E_Gaus).item()}")
        print(f"Norm2 (Gaussian Noise): {norm2_Gaus}")
        logging.info(f"Norm2 (Gaussian Noise): {norm2_Gaus}")

    E_pca_attack = None
    TN_G_pca = None
    norm_pca_attack = None
    theta_pca_attack = None
    if attack_enable_pca_subspace:
        with peak_memory_monitor("PCA_Subspace"):
            E_pca_attack, attacked_tensor_pca, theta_pca_attack = run_pca_subspace_attack(
                X,
                config.epsilon,
                pca_attack_rank,
                pca_attack_rank1_trials,
                pca_attack_full_trials,
                mode=pca_attack_mode,
                verbose=pca_attack_verbose,
                verbose_interval=pca_attack_verbose_interval,
            )
            tr_cores_pca = tr.tr_decompose2(
                attacked_tensor_pca,
                clone_tr_cores(tr_cores_init),
                max_iter=config.inner_num_steps,
                tol=inner_tol,
                device="cuda",
            )
            TN_G_pca = tl.tr_to_tensor(tr_cores_pca)
            norm_pca_attack = torch.norm(X - TN_G_pca).item()
            print(f"Norm (PCA Subspace Attack): {norm_pca_attack}, theta={theta_pca_attack:.6f}")
            logging.info(f"Norm (PCA Subspace Attack): {norm_pca_attack}, theta={theta_pca_attack:.6f}")

    attack_distribution_payloads: Dict[str, Dict[str, torch.Tensor]] = {}
    if distribution_plots_enabled:
        def register_distribution_case(label: str, noise_tensor: Optional[torch.Tensor], reconstruction_tensor: Optional[torch.Tensor]):
            if noise_tensor is None or reconstruction_tensor is None:
                return
            noise_detached = noise_tensor.detach()
            attack_distribution_payloads[label] = {
                "noise": noise_detached,
                "attacked": (X + noise_detached).detach(),
                "reconstruction": reconstruction_tensor.detach(),
            }

        if attack_enable_clean and zero_perturbation is not None and TN_G_ori is not None:
            register_distribution_case("Clean", zero_perturbation, TN_G_ori)
        if attack_enable_gaussian and E_Gaus is not None and TN_G_Gaus is not None:
            register_distribution_case("Gaussian", E_Gaus, TN_G_Gaus)
        if attack_enable_ours_f and E_optimized_F is not None and TN_G_F is not None:
            register_distribution_case("Ours_F", E_optimized_F, TN_G_F)
        if attack_enable_ours and E_optimized is not None and TN_G is not None:
            register_distribution_case("Ours", E_optimized, TN_G)
        if attack_enable_attr_attack and E_attr_attack is not None and TN_G_attr_attack is not None:
            register_distribution_case("ATTR_Attack", E_attr_attack, TN_G_attr_attack)
        if attack_enable_pca_subspace and E_pca_attack is not None and TN_G_pca is not None:
            register_distribution_case("PCA_Attack", E_pca_attack, TN_G_pca)

    principal_angle_summary: Dict[str, Dict[str, Optional[float]]] = {}
    if distribution_plots_enabled and attack_distribution_payloads:
        distribution_dir = os.path.join(output_folder, 'distribution_plots', base_name)
        os.makedirs(distribution_dir, exist_ok=True)
        input_np = tensor_to_numpy_array(X)
        input_fft_np = fft_magnitude(input_np)
        for label, payload in attack_distribution_payloads.items():
            noise_np = tensor_to_numpy_array(payload["noise"])
            attacked_np = tensor_to_numpy_array(payload["attacked"])
            recon_np = tensor_to_numpy_array(payload["reconstruction"])
            plot_attack_distribution_figures(
                label,
                base_name,
                distribution_dir,
                input_np,
                input_fft_np,
                noise_np,
                attacked_np,
                recon_np,
            )
            noise_tensor = payload["noise"]
            if noise_tensor is None or torch.norm(noise_tensor).item() <= 1e-12:
                continue
            clean_angles = []
            if 'TN_G_ori' in locals() and TN_G_ori is not None:
                clean_angles = compute_modewise_principal_angles(noise_tensor, TN_G_ori, mode_rank_products)
            attack_angles = compute_modewise_principal_angles(noise_tensor, payload["reconstruction"], mode_rank_products)
            principal_angle_summary[label] = {
                "clean_mean": _mean_angle(clean_angles),
                "attack_mean": _mean_angle(attack_angles),
            }

    if principal_angle_summary:
        print_principal_angle_table(base_name, principal_angle_summary)

    if multi_decomposition_attack_enable:
        run_multi_decomposition_attack_suite(X, config, base_name, tr_cores_init)

    if linf_attack_enable:
        run_linf_attack_experiment(
            X,
            base_name,
            config,
            tr_cores_init,
            clean_norm=norm2_original_TR_algorithm,
            clean_reconstruction=TN_G_ori,
        )

    perturbations_for_defence = []

    if attack_enable_clean and zero_perturbation is not None and norm2_original_TR_algorithm is not None:
        perturbations_for_defence.append(("Clean", zero_perturbation, norm2_original_TR_algorithm))

    if attack_enable_gaussian and E_Gaus is not None and norm2_Gaus is not None:
        perturbations_for_defence.append(("Gaussian", E_Gaus, norm2_Gaus))

    if attack_enable_ours_f and E_optimized_F is not None and norm_attack_ours_F is not None:
        perturbations_for_defence.append(("Ours_F", E_optimized_F, norm_attack_ours_F))

    if attack_enable_ours and E_optimized is not None and norm_attack_ours is not None:
        perturbations_for_defence.append(("Ours", E_optimized, norm_attack_ours))
    if attack_enable_attr_attack and E_attr_attack is not None and norm_attack_attr is not None:
        perturbations_for_defence.append(("ATTR_Attack", E_attr_attack, norm_attack_attr))
    if attack_enable_pca_subspace and E_pca_attack is not None and norm_pca_attack is not None:
        perturbations_for_defence.append(("PCA_Attack", E_pca_attack, norm_pca_attack))

    results_table = []
    attr_loss_curves: Dict[str, List[float]] = {}
    attr_dfadatr_loss_curves: Dict[str, List[float]] = {}
    df_loss_curves: Dict[str, List[float]] = {}
    df_noisy_loss_curves: Dict[str, List[float]] = {}
    theta_loss_curves: Dict[str, List[float]] = {}
    attr_reconstructions: Dict[str, torch.Tensor] = {}
    attr_perturbations: Dict[str, torch.Tensor] = {}
    theta_corrections: Dict[str, torch.Tensor] = {}
    theta_reconstructions: Dict[str, torch.Tensor] = {}
    attacked_tensors: Dict[str, torch.Tensor] = {}

    if not perturbations_for_defence:
        print('未启用任何攻击开关，跳过 Defence 评估。')

    loss_ATR = []

    for label, perturbation, baseline_tr_norm in perturbations_for_defence:
        defence_cfg = config
        if label.lower() == "ours_f":
            defence_cfg = clone_config(config)
            defence_cfg.defence_lr = 0.1
        label_cores = clone_tr_cores(tr_cores_init)

        attacked_tensor = attacked_tensors.get(label)
        if attacked_tensor is None:
            attacked_tensor = (X + perturbation).detach()
            attacked_tensors[label] = attacked_tensor

        attr_norm = None
        attr_dfadatr_norm = None
        if attr_defence_enable:
            start_time = time.time()
            E_attr, TN_G_attr, attr_losses = optimize_perturbation_comparison(
                attacked_tensor,
                clone_tr_cores(label_cores),
                defence_cfg,
                attr_epsilon=defence_cfg.attr_epsilon,
                clean_tensor=X,
            )
            attr_elapsed = time.time() - start_time
            attr_norm = torch.norm(X - TN_G_attr).item()
            attr_reconstructions[label] = TN_G_attr.detach()
            attr_loss_curves[label] = [float(val) for val in attr_losses]
            loss_ATR = attr_loss_curves[label]
            attr_perturbations[label] = E_attr.clone()
            logging.info(f"ATTR Defence ({label}) -> norm: {attr_norm}, time: {attr_elapsed:.4f}s")
            if lr_finder_enabled and lr_finder_values and lr_finder_output_dir:
                _run_attr_lr_finder(
                    X,
                    perturbation,
                    tr_cores_init,
                    defence_cfg,
                    lr_finder_values,
                    base_name,
                    label,
                    lr_finder_output_dir,
                )

        theta_norm = None
        if theta_defence_enable:
            start_time_theta = time.time()
            Theta_opt, TN_G_theta, theta_losses = optimize_theta_defence(
                X,
                attacked_tensor,
                clone_tr_cores(label_cores),
                defence_cfg,
            )
            theta_elapsed = time.time() - start_time_theta
            theta_norm = torch.norm(X - TN_G_theta).item()
            theta_loss_curves[label] = [float(val) for val in theta_losses]
            theta_corrections[label] = Theta_opt.clone()
            theta_reconstructions[label] = TN_G_theta.detach()
            logging.info(
                f"Theta Defence ({label}) -> norm: {theta_norm}, ||Theta||={torch.norm(Theta_opt).item():.6f}, "
                f"time: {theta_elapsed:.4f}s"
            )

        df_noisy_norm = None
        df_noisy_norm_attack = None
        if df_adatr_noisy_enable:
            optimized_G_list_noisy, loss_list_noisy = optimize_G_list_noisy(
                X,
                clone_tr_cores(label_cores),
                perturbation,
                defence_cfg,
            )
            df_noisy_loss_curves[label] = [float(val) for val in loss_list_noisy]
            with torch.no_grad():
                _, TN_G_noisy, _ = objective_function_DFTR_noisy(
                    X,
                    [core.clone().detach() for core in optimized_G_list_noisy],
                    perturbation,
                    inner_num_steps=defence_cfg.inner_num_steps,
                    inner_tol=defence_cfg.inner_tol,
                )
            df_noisy_norm_attack = torch.norm(attacked_tensor - TN_G_noisy).item()
            df_noisy_norm = torch.norm(X - TN_G_noisy).item()
            print(
                f"Norm (DF-ADdaTR-Noisy Defence - {label}): clean={df_noisy_norm}, "
                f"attack={df_noisy_norm_attack}"
            )
            logging.info(
                f"Norm (DF-ADdaTR-Noisy Defence - {label}): clean={df_noisy_norm}, "
                f"attack={df_noisy_norm_attack}"
            )

        df_adatr_norm = None
        optimized_G_list = None
        need_df_cores = df_adatr_enable or attr_dfadatr_enable
        if need_df_cores:
            optimized_G_list, loss_list = optimize_G_list(X, clone_tr_cores(tr_cores_init), perturbation, defence_cfg)
            df_loss_curves[label] = [float(val) for val in loss_list]
            G_list_init = [core.clone().detach() for core in optimized_G_list]
            if df_adatr_enable:
                with torch.no_grad():
                    eval_loss, TN_G_eval, _ = objective_function_DFTR(
                        X,
                        G_list_init,
                        perturbation,
                        inner_num_steps=defence_cfg.inner_num_steps,
                        inner_tol=defence_cfg.inner_tol,
                    )
                df_adatr_norm = torch.norm(X - TN_G_eval).item()
                print(f"Norm (DF-ADdaTR Defence - {label}): {df_adatr_norm}")
                logging.info(f"Norm (DF-ADdaTR Defence - {label}): {df_adatr_norm}")
            if lr_finder_enabled and lr_finder_values and lr_finder_output_dir and df_adatr_enable:
                _run_df_lr_finder(
                    X,
                    perturbation,
                    tr_cores_init,
                    defence_cfg,
                    lr_finder_values,
                    base_name,
                    label,
                    lr_finder_output_dir,
                )
            if attr_dfadatr_enable:
                attacked_tensor_df = (X + perturbation).detach()
                start_time_df = time.time()
                E_attr_df, TN_G_attr_df, attr_losses_df = optimize_perturbation_comparison(
                    attacked_tensor_df,
                    [core.clone().detach() for core in optimized_G_list],
                    defence_cfg,
                    attr_epsilon=defence_cfg.attr_epsilon,
                    clean_tensor=X,
                )
                attr_dfadatr_norm = torch.norm(X - TN_G_attr_df).item()
                attr_dfadatr_loss_curves[label] = [float(val) for val in attr_losses_df]
                logging.info(
                    f"ATTR(DF-ADdaTR init) Defence ({label}) -> norm: {attr_dfadatr_norm}, "
                    f"time: {time.time() - start_time_df:.4f}s"
                )
                if lr_finder_enabled and lr_finder_values and lr_finder_output_dir:
                    _run_attr_df_lr_finder(
                        X,
                        perturbation,
                        tr_cores_init,
                        defence_cfg,
                        lr_finder_values,
                        base_name,
                        label,
                        lr_finder_output_dir,
                    )

        ata_norm = None
        if ata_defence_enable:
            attack_cfg = clone_config(config)
            defence_cfg_ata = clone_config(defence_cfg)
            label_lower = label.lower()
            if label_lower == "clean":
                ata_type = "clean"
            elif label_lower == "gaussian":
                ata_type = "gaussian"
            elif label_lower == "ours_f":
                ata_type = "simplified"
            else:
                ata_type = "learned"
            ata_result = alternating_attack_training(
                X,
                [core.clone().detach() for core in tr_cores],
                attack_cfg,
                defence_cfg_ata,
                num_rounds=config.outer_num_steps_defence,
                track_history=False,
                store_perturbations=False,
                base_perturbation=perturbation,
                perturbation_type=ata_type,
            )
            ata_best = ata_result['best_G_list']
            ata_norm = _evaluate_defence(X, ata_best, perturbation, config)
            print(f"Norm (ATAdaTR Defence - {label}): {ata_norm}")
            logging.info(f"Norm (ATAdaTR Defence - {label}): {ata_norm}")
            if lr_finder_enabled and lr_finder_values and lr_finder_output_dir:
                _run_ata_lr_finder(
                    X,
                    perturbation,
                    tr_cores_init,
                    attack_cfg,
                    defence_cfg,
                    lr_finder_values,
                    base_name,
                    label,
                    lr_finder_output_dir,
                )

        results_entry = {
            "label": label,
            "tr_norm": baseline_tr_norm,
        }
        if theta_defence_enable:
            results_entry["theta_norm"] = theta_norm
        if df_adatr_noisy_enable:
            results_entry["df_adatr_noisy_norm"] = df_noisy_norm
        if attr_defence_enable:
            results_entry["attr_norm"] = attr_norm
        if attr_dfadatr_enable:
            results_entry["attr_dfadatr_norm"] = attr_dfadatr_norm
        if df_adatr_enable:
            results_entry["df_adatr_norm"] = df_adatr_norm
        if ata_defence_enable:
            results_entry["ata_norm"] = ata_norm
        results_table.append(results_entry)

    if results_table:
        columns = [("TR Norm", "tr_norm", 15)]
        if theta_defence_enable:
            columns.append(("Theta Norm", "theta_norm", 15))
        if df_adatr_noisy_enable:
            columns.append(("DF-ADdaTR(Noisy) Clean", "df_adatr_noisy_norm", 24))
        if attr_defence_enable:
            columns.append(("ATTR Norm", "attr_norm", 15))
        if attr_dfadatr_enable:
            columns.append(("ATTR(DF-ADdaTR) Norm", "attr_dfadatr_norm", 22))
        if df_adatr_enable:
            columns.append(("DF-ADdaTR Norm", "df_adatr_norm", 18))
        if ata_defence_enable:
            columns.append(("ATAdaTR Norm", "ata_norm", 18))

        header = f"{'Case':<12}" + "".join(f"{title:>{width}}" for title, _, width in columns)
        print(header)
        logging.info(header)
        print('-' * len(header))
        logging.info('-' * len(header))
        for row in results_table:
            line = f"{row['label']:<12}"
            log_parts = [f"TR Norm: {row['tr_norm']:.6f}"]
            for title, key, width in columns:
                value = row.get(key)
                if key == "tr_norm":
                    formatted = f"{value:>{width}.6f}"
                else:
                    formatted = f"{value:>{width}.6f}" if value is not None else f"{'--':>{width}}"
                if key != "tr_norm":
                    log_val = f"{value:.6f}" if value is not None else '--'
                    log_parts.append(f"{title}: {log_val}")
                line += formatted
            print(line)
            logging.info(line)
            logging.info(f"Table Row -> Case: {row['label']}, " + ", ".join(log_parts))
        record_image_summary(base_name, results_table)

    if plot_losses_enabled:
        loss_plot_folder = os.path.join(output_folder, 'loss_plots')
        os.makedirs(loss_plot_folder, exist_ok=True)

        if norm2_original_TR_algorithm is not None:
            plot_loss_curves(
                {},
                f'Baseline TR Reconstruction ({base_name})',
                '||X - TN||',
                os.path.join(loss_plot_folder, f'{base_name}_baseline.png'),
                baseline=float(norm2_original_TR_algorithm),
            )

        if attr_loss_curves:
            plot_loss_curves(
                {f'ATTR-{lbl}': vals for lbl, vals in attr_loss_curves.items()},
                f'ATTR Loss Curves ({base_name})',
                'Reconstruction Error',
                os.path.join(loss_plot_folder, f'{base_name}_attr.png'),
            )

        if attr_dfadatr_loss_curves:
            plot_loss_curves(
                {f'ATTR(DF)-{lbl}': vals for lbl, vals in attr_dfadatr_loss_curves.items()},
                f'ATTR w/ DF-ADdaTR Init ({base_name})',
                'Reconstruction Error',
                os.path.join(loss_plot_folder, f'{base_name}_attr_df.png'),
            )

        if df_loss_curves:
            plot_loss_curves(
                {f'DF-ADdaTR-{lbl}': vals for lbl, vals in df_loss_curves.items()},
                f'DF-ADdaTR Loss Curves ({base_name})',
                'Objective',
                os.path.join(loss_plot_folder, f'{base_name}_df_adatr.png'),
            )

        if df_noisy_loss_curves:
            plot_loss_curves(
                {f'DF-ADdaTR-Noisy-{lbl}': vals for lbl, vals in df_noisy_loss_curves.items()},
                f'DF-ADdaTR (Noisy) Loss Curves ({base_name})',
                'Objective',
                os.path.join(loss_plot_folder, f'{base_name}_df_adatr_noisy.png'),
            )

        if theta_loss_curves:
            plot_loss_curves(
                {f'Theta-{lbl}': vals for lbl, vals in theta_loss_curves.items()},
                f'Theta Defence Loss Curves ({base_name})',
                'Reconstruction Error',
                os.path.join(loss_plot_folder, f'{base_name}_theta.png'),
            )

        combined_curves: Dict[str, List[float]] = {}
        combined_curves.update({f'ATTR-{lbl}': vals for lbl, vals in attr_loss_curves.items()})
        combined_curves.update({f'ATTR(DF)-{lbl}': vals for lbl, vals in attr_dfadatr_loss_curves.items()})
        combined_curves.update({f'DF-ADdaTR-{lbl}': vals for lbl, vals in df_loss_curves.items()})
        combined_curves.update({f'DF-ADdaTR-Noisy-{lbl}': vals for lbl, vals in df_noisy_loss_curves.items()})
        combined_curves.update({f'Theta-{lbl}': vals for lbl, vals in theta_loss_curves.items()})
        if combined_curves or norm2_original_TR_algorithm is not None:
            plot_loss_curves(
                combined_curves,
                f'Combined Defence Losses ({base_name})',
                'Loss / Error',
                os.path.join(loss_plot_folder, f'{base_name}_combined.png'),
                baseline=float(norm2_original_TR_algorithm) if norm2_original_TR_algorithm is not None else None,
            )

    attr_vis_label = next(iter(attr_reconstructions), None)
    if attr_vis_label is not None:
        attacked_cpu = attacked_tensors[attr_vis_label].cpu().detach()
        attr_cpu = attr_reconstructions[attr_vis_label].cpu().detach()
        clean_cpu = X.cpu().detach()
        attr_noise = attr_perturbations.get(attr_vis_label)
        if attr_noise is not None:
            noise_cpu = attr_noise.cpu().detach()
        else:
            noise_cpu = attacked_cpu - clean_cpu
        E_comparison_cpu = noise_cpu.numpy()
        X_E_comparison_cpu = (attacked_cpu + noise_cpu).numpy()
        TN_G_comparison = attr_cpu.numpy()
    else:
        E_comparison_cpu = None
        X_E_comparison_cpu = None
        TN_G_comparison = None

    if adveresarial_training_enable:
        print('\n=== Adversarial ATTR Summary ===')
        adv_summary = run_adversarial_comparison(
            X,
            [core.clone().detach() for core in tr_cores],
            config,
            config,
            adv_rounds= 100,
            track_history=False,
            store_perturbations=False,
            print_table=True,
        )
        logging.info(f"Adversarial comparison summary: {adv_summary}")

    imshow_requirements_met = attack_enable_clean and attack_enable_gaussian and attack_enable_ours and attack_enable_ours_f

    if imshow_enabled:
        if not imshow_requirements_met:
            print('imshow_enabled 为 True，但部分攻击开关关闭，跳过可视化。')
        else:
            # 将张量从 CUDA 移动到 CPU 并转换为 NumPy 数组
            X_cpu = X.cpu().detach().numpy()
            E_cpu = E_optimized.cpu().detach().numpy()
            E_cpu_F = E_optimized_F.cpu().detach().numpy()
            E_cpu_Gaus = E_Gaus.cpu().detach().numpy()
            X_E_cpu = (X + E_optimized).cpu().detach().numpy()
            X_E_cpu_F = (X + E_optimized_F).cpu().detach().numpy()
            X_E_cpu_Gaus = (X + E_Gaus).cpu().detach().numpy()
            TN_G = (TN_G).cpu().detach().numpy()
            TN_G_F = (TN_G_F).cpu().detach().numpy()
            TN_G_Gaus = (TN_G_Gaus).cpu().detach().numpy()
            TN_G_ori = TN_G_ori.cpu().detach().numpy()

            # 显示 TN_G 图像
            plt.clf()
            plt.imshow(X_E_cpu)
            save_mat_file(output_folder, f'X_E_Ours_{base_name}', X_E_cpu)

            plt.axis('off')
            fig = plt.gcf()  # 获取当前图形对象
            save_fig(output_folder, f'X_E_Ours_{base_name}', fig)  # 保存图形

            plt.imshow(X_E_cpu_F)
            save_mat_file(output_folder, f'X_E_Ours_F_{base_name}', X_E_cpu_F)
            plt.axis('off')
            fig = plt.gcf()  # 获取当前图形对象
            save_fig(output_folder, f'X_E_Ours_F_{base_name}', fig)  # 保存图形

            # 显示 TN_G 图像
            plt.imshow(X_cpu)
            save_mat_file(output_folder, f'X_{base_name}', X_cpu)
            plt.axis('off')

            fig = plt.gcf()  # 获取当前图形对象
            save_fig(output_folder, f'X_{base_name}', fig)  # 保存图形

            plt.clf()
            plt.imshow(X_E_cpu_Gaus)
            save_mat_file(output_folder, f'X_E_Gaus_{base_name}', X_E_cpu_Gaus)
            plt.axis('off')

            fig = plt.gcf()  # 获取当前图形对象
            save_fig(output_folder, f'X_E_Gaus_{base_name}', fig)  # 保存图形

    save_path = os.path.join(output_folder, f'comparison_{base_name}.png')

    if attr_defence_enable and imshow_enabled and imshow_requirements_met and attr_vis_label is not None:
        
        # 显示 TN_G 图像
        plt.imshow(X_E_comparison_cpu)
        save_mat_file(output_folder, f'X_E_ATTR_{base_name}', X_E_comparison_cpu)
        plt.axis('off')

        # 保存图像
        fig = plt.gcf()  # 获取当前图形对象
        save_fig(output_folder, f'X_E_ATTR_{base_name}', fig)  # 保存图形

        # 创建一个全黑的图像
        black_image = np.zeros_like(X_cpu)

        images = [
        X_cpu, black_image, X_cpu, TN_G_ori,  # 第一行
        X_cpu, E_cpu_Gaus, X_E_cpu_Gaus, TN_G_Gaus,  # 第二行
        X_cpu, E_comparison_cpu, X_E_comparison_cpu, TN_G_comparison,  # 第三行
        X_cpu, E_cpu_F, X_E_cpu_F, TN_G_F,  # 第三行
        X_cpu, E_cpu, X_E_cpu, TN_G      # 第二行
        ]   

        row_titles = ['Clean', 'Guassian', 'ATTR', 'Ours_F', 'Ours']  # 行标题列表
        col_titles = ['Original Image', 'Noise Image', 'Ori + Noise', 'Reconstructed Image']  # 列标题列表
        display_images(X_cpu, images, row_titles, col_titles, 5, 4, save_path=save_path)


        # 创建一个包含三个子图的图形
        plt.figure(figsize=(15, 10))

        # 第一个子图：同时显示 loss_ours 和 loss_ATR
        plt.subplot(4, 1, 1)
        plt.plot(loss_ours, label='Ours')
        plt.plot(loss_ours_F, label='Ours_F')
        plt.plot(loss_ATR, label='ATTR')
        plt.xlabel('Step')
        plt.ylabel('Loss (sqrt)')
        plt.title('Loss over Steps')
        plt.legend()
        plt.grid(True)

        # 第二个子图：显示 loss_ours
        plt.subplot(4, 1, 2)
        plt.plot(loss_ours, label='Ours', color='blue')
        plt.xlabel('Step')
        plt.ylabel('Loss (sqrt)')
        plt.title('Loss over Steps (Ours)')
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 3)
        plt.plot(loss_ours_F, label='Ours_F', color='orange')
        plt.xlabel('Step')
        plt.ylabel('Loss (sqrt)')
        plt.title('Loss over Steps (Ours_F)')
        plt.legend()
        plt.grid(True)

        # 第三个子图：显示 loss_ATR
        plt.subplot(4, 1, 4)
        plt.plot(loss_ATR, label='ATTR', color='green')
        plt.xlabel('Step')
        plt.ylabel('Loss (sqrt)')
        plt.title('Loss over Steps (ATTR)')
        plt.legend()
        plt.grid(True)

        # 调整子图布局
        plt.tight_layout()

        # 保存图形
        if save_fig_enabled:
            plt.savefig(f'{output_folder}/loss_plot_{base_name}.png')

        # 显示图形
        plt.show()

        # 绘制柱状图
        labels = ['simplified', 'optimized', 'comparison']
        times = [time_simplified, time_optimized, time_comp]
        plt.figure(figsize=(10, 5))
        plt.bar(labels, times, color=['blue', 'green', 'red'])
        plt.xlabel('Function')
        plt.ylabel('Time (seconds)')
        plt.title('Function Execution Time Comparison')
        if save_fig_enabled:
            plt.savefig(f'{output_folder}/function_time_comparison_{base_name}.png')
        plt.show()

    print_peak_memory_summary(peak_memory_records)
    set_peak_memory_recorder(None)

    return time_simplified, time_optimized


def process_image(image_path, output_folder, config, resize_shape=None, resize_width=None):
    image = Image.open(image_path).convert('RGB')
    if resize_shape is not None:
        image = image.resize(resize_shape, Image.BILINEAR)
    elif resize_width is not None:
        orig_width, orig_height = image.size
        if orig_width == 0:
            raise ValueError("Image width is zero, cannot resize proportionally.")
        scale = resize_width / orig_width
        new_height = max(1, int(round(orig_height * scale)))
        image = image.resize((resize_width, new_height), Image.BILINEAR)
    image = np.array(image)
    X = torch.tensor(image / 255.0, dtype=torch.float32)
    base_name = os.path.basename(image_path).split('.')[0]
    return process_tensor_data(X, base_name, output_folder, config)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial Tensor Ring experiments on images, synthetic tensors, or videos.")
    parser.add_argument(
        "--data-source",
        choices=["image", "synthetic", "video"],
        default="image",
        help="Choose whether to process image data from disk or generate synthetic low-rank tensor data.",
    )
    parser.add_argument(
        "--input-folder",
        default="./AT_TD/My picture",
        help="Folder containing input images when --data-source=image.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Starting index (inclusive) for image filenames like 1.png, 2.png, ...",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=1,
        help="Ending index (inclusive) for image filenames when processing images.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Optional resize (width height) applied to each input image before processing.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=None,
        help="Resize images to this width while preserving aspect ratio (height scales accordingly).",
    )
    parser.add_argument(
        "--video-dir",
        default="./AT_TD/video_data",
        help="Directory containing .mat video tensors when --data-source=video.",
    )
    parser.add_argument(
        "--video-name",
        default=None,
        help="Specific video filename to process (within --video-dir). If omitted, processes the first video unless --video-all is set.",
    )
    parser.add_argument(
        "--video-all",
        action="store_true",
        help="Process every .mat video found in --video-dir.",
    )
    parser.add_argument(
        "--video-resize",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Resize each video frame to the specified width and height before processing.",
    )
    parser.add_argument(
        "--min-video-frames",
        type=int,
        default=20,
        help="Minimum number of frames to use from each video (will warn if fewer are available).",
    )
    parser.add_argument(
        "--max-video-frames",
        type=int,
        default=30,
        help="Maximum number of frames to use from each video.",
    )
    parser.add_argument(
        "--attr-defence",
        dest="attr_defence_enable",
        action="store_true",
        help="Enable ATTR defence evaluation.",
    )
    parser.add_argument(
        "--no-attr-defence",
        dest="attr_defence_enable",
        action="store_false",
        help="Disable ATTR defence evaluation.",
    )
    parser.add_argument(
        "--attr-df-adatr",
        dest="attr_dfadatr_enable",
        action="store_true",
        help="Enable ATTR defence initialised with DF-ADdaTR cores.",
    )
    parser.add_argument(
        "--no-attr-df-adatr",
        dest="attr_dfadatr_enable",
        action="store_false",
        help="Disable ATTR defence initialised with DF-ADdaTR cores.",
    )
    parser.add_argument(
        "--df-adatr",
        dest="df_adatr_enable",
        action="store_true",
        help="Enable DF-ADdaTR defence evaluation.",
    )
    parser.add_argument(
        "--no-df-adatr",
        dest="df_adatr_enable",
        action="store_false",
        help="Disable DF-ADdaTR defence evaluation.",
    )
    parser.add_argument(
        "--df-noisy-defence",
        dest="df_adatr_noisy_enable",
        action="store_true",
        help="Enable DF-ADdaTR defence that fits the noisy tensor.",
    )
    parser.add_argument(
        "--no-df-noisy-defence",
        dest="df_adatr_noisy_enable",
        action="store_false",
        help="Disable DF-ADdaTR defence that fits the noisy tensor.",
    )
    parser.add_argument(
        "--multi-attack",
        dest="multi_decomposition_attack_enable",
        action="store_true",
        help="Enable multi-decomposition attack experiments (TR/TT/Tucker/CP).",
    )
    parser.add_argument(
        "--no-multi-attack",
        dest="multi_decomposition_attack_enable",
        action="store_false",
        help="Disable multi-decomposition attack experiments.",
    )
    parser.add_argument(
        "--multi-attack-tr",
        dest="attack_multi_tr_enable",
        action="store_true",
        help="Include TR decomposition in multi-attack experiment.",
    )
    parser.add_argument(
        "--no-multi-attack-tr",
        dest="attack_multi_tr_enable",
        action="store_false",
        help="Exclude TR decomposition from multi-attack experiment.",
    )
    parser.add_argument(
        "--multi-attack-tt",
        dest="attack_multi_tt_enable",
        action="store_true",
        help="Include TT decomposition attack.",
    )
    parser.add_argument(
        "--no-multi-attack-tt",
        dest="attack_multi_tt_enable",
        action="store_false",
        help="Exclude TT decomposition attack.",
    )
    parser.add_argument(
        "--multi-attack-tucker",
        dest="attack_multi_tucker_enable",
        action="store_true",
        help="Include Tucker decomposition attack.",
    )
    parser.add_argument(
        "--no-multi-attack-tucker",
        dest="attack_multi_tucker_enable",
        action="store_false",
        help="Exclude Tucker decomposition attack.",
    )
    parser.add_argument(
        "--multi-attack-cp",
        dest="attack_multi_cp_enable",
        action="store_true",
        help="Include CP decomposition attack.",
    )
    parser.add_argument(
        "--no-multi-attack-cp",
        dest="attack_multi_cp_enable",
        action="store_false",
        help="Exclude CP decomposition attack.",
    )
    parser.add_argument(
        "--multi-tt-rank",
        type=int,
        default=tt_attack_rank,
        help="TT rank (per bond) for multi-decomposition attacks.",
    )
    parser.add_argument(
        "--multi-tucker-rank",
        type=int,
        default=tucker_attack_rank,
        help="Tucker rank (per mode) for multi-decomposition attacks.",
    )
    parser.add_argument(
        "--multi-cp-rank",
        type=int,
        default=cp_attack_rank,
        help="CP rank for multi-decomposition attacks.",
    )
    parser.add_argument(
        "--multi-attack-iters",
        type=int,
        default=multi_attack_inner_iters,
        help="Number of ALS iterations for each decomposition during multi-attack experiment.",
    )
    parser.add_argument(
        "--linf-attack",
        dest="linf_attack_enable",
        action="store_true",
        help="Enable AdaTR experiment with L∞-norm constraint on E.",
    )
    parser.add_argument(
        "--no-linf-attack",
        dest="linf_attack_enable",
        action="store_false",
        help="Disable AdaTR L∞ experiment.",
    )
    parser.add_argument(
        "--pca-attack",
        dest="attack_enable_pca_subspace",
        action="store_true",
        help="Enable PCA-based subspace perturbation attack (first-mode unfolding).",
    )
    parser.add_argument(
        "--no-pca-attack",
        dest="attack_enable_pca_subspace",
        action="store_false",
        help="Disable PCA-based subspace perturbation attack.",
    )
    parser.add_argument(
        "--pca-attack-rank",
        type=int,
        default=pca_attack_rank,
        help="Target PCA subspace rank k for the subspace attack.",
    )
    parser.add_argument(
        "--pca-rank1-trials",
        type=int,
        default=pca_attack_rank1_trials,
        help="Number of Monte Carlo trials for rank-one PCA attacks.",
    )
    parser.add_argument(
        "--pca-full-trials",
        type=int,
        default=pca_attack_full_trials,
        help="Number of Monte Carlo trials for full-matrix PCA attacks.",
    )
    parser.add_argument(
        "--pca-attack-mode",
        choices=["rank1", "full", "both"],
        default=pca_attack_mode,
        help="Sampling strategy for PCA attack (rank-one, full, or both).",
    )
    parser.add_argument(
        "--pca-verbose",
        dest="pca_attack_verbose",
        action="store_true",
        help="Print intermediate PCA attack progress (theta values).",
    )
    parser.add_argument(
        "--no-pca-verbose",
        dest="pca_attack_verbose",
        action="store_false",
        help="Disable PCA attack progress prints.",
    )
    parser.add_argument(
        "--pca-verbose-interval",
        type=int,
        default=pca_attack_verbose_interval,
        help="How often (in iterations) to print PCA attack progress when verbose is enabled.",
    )
    parser.add_argument(
        "--plot-distributions",
        dest="distribution_plots_enabled",
        action="store_true",
        help="Enable plotting spatial/Fourier value distributions for every attack case.",
    )
    parser.add_argument(
        "--no-plot-distributions",
        dest="distribution_plots_enabled",
        action="store_false",
        help="Disable plotting spatial/Fourier value distributions.",
    )
    parser.add_argument(
        "--ata-defence",
        dest="ata_defence_enable",
        action="store_true",
        help="Enable alternating attack (ATAdaTR) defence evaluation.",
    )
    parser.add_argument(
        "--no-ata-defence",
        dest="ata_defence_enable",
        action="store_false",
        help="Disable alternating attack (ATAdaTR) defence evaluation.",
    )
    parser.add_argument(
        "--theta-defence",
        dest="theta_defence_enable",
        action="store_true",
        help="Enable Theta-based defence optimisation.",
    )
    parser.add_argument(
        "--no-theta-defence",
        dest="theta_defence_enable",
        action="store_false",
        help="Disable Theta-based defence optimisation.",
    )
    parser.add_argument(
        "--synthetic-shape",
        type=int,
        nargs=3,
        default=(50, 50, 50),
        help="Shape of the generated synthetic tensor.",
    )
    parser.add_argument(
        "--synthetic-rank",
        type=int,
        nargs=3,
        default=(6, 6, 6),
        help="Tensor Ring ranks for each mode of the synthetic tensor.",
    )
    parser.add_argument(
        "--synthetic-seed",
        type=int,
        default=None,
        help="Optional random seed for synthetic tensor generation.",
    )
    parser.add_argument(
        "--attr-epsilon",
        type=float,
        default=0.1,
        help="Maximum Frobenius norm allowed for ATTR defence scaling (set smaller than attack epsilon).",
    )
    parser.add_argument(
        "--theta-epsilon",
        type=float,
        default=None,
        help="Maximum Frobenius norm allowed for Theta defence corrections (defaults to ATTR epsilon).",
    )
    parser.add_argument(
        "--attack-lr",
        type=float,
        default=0.1,
        help="Learning rate for attack optimizers.",
    )
    parser.add_argument(
        "--defence-lr",
        type=float,
        default=0.01,
        help="Learning rate for defence optimizers.",
    )
    parser.add_argument(
        "--outer-tol",
        type=float,
        default=1e-6,
        help="Tolerance for change in loss to early-stop outer optimisation loops.",
    )
    parser.set_defaults(
        attr_defence_enable=attr_defence_enable,
        attr_dfadatr_enable=attr_dfadatr_enable,
        df_adatr_enable=df_adatr_enable,
        df_adatr_noisy_enable=df_adatr_noisy_enable,
        ata_defence_enable=ata_defence_enable,
        theta_defence_enable=theta_defence_enable,
        distribution_plots_enabled=distribution_plots_enabled,
        multi_decomposition_attack_enable=multi_decomposition_attack_enable,
        attack_multi_tr_enable=attack_multi_tr_enable,
        attack_multi_tt_enable=attack_multi_tt_enable,
        attack_multi_tucker_enable=attack_multi_tucker_enable,
        attack_multi_cp_enable=attack_multi_cp_enable,
        linf_attack_enable=linf_attack_enable,
        attack_enable_pca_subspace=attack_enable_pca_subspace,
        pca_attack_verbose=pca_attack_verbose,
    )
    args = parser.parse_args()

    input_folder = args.input_folder
    resize_shape = tuple(args.resize) if args.resize else None
    resize_width = args.resize_width
    video_dir = args.video_dir
    video_name = args.video_name
    video_all = args.video_all
    video_resize = tuple(args.video_resize) if args.video_resize else None
    min_video_frames = args.min_video_frames
    max_video_frames = args.max_video_frames
    attr_defence_enable = args.attr_defence_enable
    attr_dfadatr_enable = args.attr_dfadatr_enable
    df_adatr_enable = args.df_adatr_enable
    df_adatr_noisy_enable = args.df_adatr_noisy_enable
    ata_defence_enable = args.ata_defence_enable
    theta_defence_enable = args.theta_defence_enable
    distribution_plots_enabled = args.distribution_plots_enabled
    multi_decomposition_attack_enable = args.multi_decomposition_attack_enable
    attack_multi_tr_enable = args.attack_multi_tr_enable
    attack_multi_tt_enable = args.attack_multi_tt_enable
    attack_multi_tucker_enable = args.attack_multi_tucker_enable
    attack_multi_cp_enable = args.attack_multi_cp_enable
    linf_attack_enable = args.linf_attack_enable
    attack_enable_pca_subspace = args.attack_enable_pca_subspace
    pca_attack_rank = args.pca_attack_rank
    pca_attack_rank1_trials = args.pca_rank1_trials
    pca_attack_full_trials = args.pca_full_trials
    pca_attack_mode = args.pca_attack_mode
    pca_attack_verbose = args.pca_attack_verbose
    pca_attack_verbose_interval = args.pca_verbose_interval
    tt_attack_rank = args.multi_tt_rank
    tucker_attack_rank = args.multi_tucker_rank
    cp_attack_rank = args.multi_cp_rank
    multi_attack_inner_iters = args.multi_attack_iters
    output_folder = os.path.join(log_folder, 'output')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # epsilon_per_entry = 0.005
    epsilon_per_entry = 0.0005
    inner_tol = 1e-10
    attack_lr = args.attack_lr
    defence_lr = args.defence_lr
    outer_tol = args.outer_tol
    outer_steps_attack_default = 100
    outer_steps_defence_default = 10
    inner_num_steps = 100
    outer_dis_num = 1
    inner_dis_num = 2000
    logging.info("Optimization parameters:")
    logging.info(f"epsilon_per_entry (Mean square perturbation per entry): {epsilon_per_entry}")
    logging.info(f"inner_tol (Tolerance for inner optimization): {inner_tol}")
    logging.info(f"attack_lr (Learning rate for attacks): {attack_lr}")
    logging.info(f"defence_lr (Learning rate for defences): {defence_lr}")
    logging.info(f"outer_tol (Outer optimisation tolerance): {outer_tol}")
    outer_steps_attack = outer_steps_attack_default
    outer_steps_defence = outer_steps_defence_default
    logging.info(
        f"outer_num_steps_attack (Number of attack outer iterations): {outer_steps_attack}"
    )
    logging.info(
        f"outer_num_steps_defence (Number of defence outer iterations): {outer_steps_defence}"
    )
    logging.info(f"inner_num_steps (Number of inner optimization steps): {inner_num_steps}")
    logging.info(f"outer_dis_num (Display loss every outer_dis_num steps): {outer_dis_num}")
    logging.info(f"inner_dis_num (Display loss every inner_dis_num steps): {inner_dis_num}")
    logging.info(f"attr_epsilon (ATTR defence constraint): {args.attr_epsilon}")
    logging.info(f"theta_epsilon (Theta defence constraint): {args.theta_epsilon}")

    config = OptimizationConfig(
        0.0,
        attack_lr,
        outer_steps_attack,
        outer_steps_defence,
        inner_num_steps,
        outer_dis_num,
        inner_dis_num,
        inner_tol,
        attr_epsilon=args.attr_epsilon,
        unroll_steps=inner_num_steps,
        attack_lr=attack_lr,
        defence_lr=defence_lr,
        epsilon_per_entry=epsilon_per_entry,
        theta_epsilon=args.theta_epsilon,
        outer_tol=outer_tol,
    )
    times_simplified: List[float] = []
    times_optimized: List[float] = []

    if args.data_source == "synthetic":
        synthetic_shape = tuple(args.synthetic_shape)
        synthetic_rank = tuple(args.synthetic_rank)
        if len(synthetic_shape) != len(synthetic_rank):
            raise ValueError("synthetic_shape and synthetic_rank must have the same length")

        synthetic_tensor = generate_low_rank_tensor(
            shape=synthetic_shape,
            ranks=synthetic_rank,
            device="cuda",
            normalize=True,
            seed=args.synthetic_seed,
        )
        base_name = f"synthetic_{'x'.join(str(dim) for dim in synthetic_shape)}_r{'_'.join(str(r) for r in synthetic_rank)}"
        time_simplified, time_optimized = process_tensor_data(synthetic_tensor, base_name, output_folder, config)
        if time_simplified is not None:
            times_simplified.append(time_simplified)
        if time_optimized is not None:
            times_optimized.append(time_optimized)
    elif args.data_source == "video":
        video_dir = os.path.abspath(video_dir)
        pattern_lower = os.path.join(video_dir, "*.mat")
        pattern_upper = os.path.join(video_dir, "*.MAT")
        pattern_mixed = os.path.join(video_dir, "*.Mat")
        video_files = sorted(set(glob.glob(pattern_lower) + glob.glob(pattern_upper) + glob.glob(pattern_mixed)))
        if not video_files:
            raise ValueError(f"No .mat video files found in {video_dir}")

        if video_all:
            selected_videos = video_files
        elif video_name is not None:
            candidate = os.path.join(video_dir, video_name)
            if not os.path.isfile(candidate):
                raise FileNotFoundError(f"Specified video not found: {candidate}")
            selected_videos = [candidate]
        else:
            selected_videos = [video_files[0]]

        for video_path in selected_videos:
            try:
                video_tensor = load_video_tensor(
                    video_path,
                    min_frames=min_video_frames,
                    max_frames=max_video_frames,
                    target_size=video_resize,
                )
            except Exception as exc:
                print(f"Failed to load video {video_path}: {exc}")
                logging.exception(f"Failed to load video {video_path}: {exc}")
                continue

            base_name = os.path.splitext(os.path.basename(video_path))[0]
            time_simplified, time_optimized = process_tensor_data(
                video_tensor,
                base_name,
                output_folder,
                config,
            )
            if time_simplified is not None:
                times_simplified.append(time_simplified)
            if time_optimized is not None:
                times_optimized.append(time_optimized)
    else:
        start_idx = args.start_index
        end_idx = args.end_index
        if end_idx < start_idx:
            raise ValueError("--end-index must be greater than or equal to --start-index")

        for idx in range(start_idx, end_idx + 1):
            image_name = f'{idx}.png'
            image_path = os.path.join(input_folder, image_name)
            if not os.path.isfile(image_path):
                print(f"Image not found, skipping: {image_path}")
                continue

            time_simplified, time_optimized = process_image(
                image_path,
                output_folder,
                config,
                resize_shape=resize_shape,
                resize_width=resize_width if resize_shape is None else None,
            )
            if time_simplified is not None:
                times_simplified.append(time_simplified)
            if time_optimized is not None:
                times_optimized.append(time_optimized)

    if times_simplified:
        mean_simplified = float(np.mean(times_simplified))
        std_simplified = float(np.std(times_simplified))
        print(f"Simplified Algorithm: Mean = {mean_simplified:.4f}, Std = {std_simplified:.4f}")
    else:
        print("Simplified Algorithm: no runs recorded.")

    if times_optimized:
        mean_optimized = float(np.mean(times_optimized))
        std_optimized = float(np.std(times_optimized))
        print(f"Optimized Algorithm: Mean = {mean_optimized:.4f}, Std = {std_optimized:.4f}")
    else:
        print("Optimized Algorithm: no runs recorded.")

    print_global_image_summary()

    if multi_decomposition_attack_enable:
        print_multi_attack_global_summary(output_folder)
    if linf_attack_enable:
        print_linf_attack_global_summary(output_folder)
