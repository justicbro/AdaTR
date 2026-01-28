import argparse
import torch
from typing import Any, Dict, List, Optional
import tensorly as tl
import torch.autograd as autograd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tensorly.decomposition import tensor_train
import TR_func as tr
from datetime import datetime
import logging
import os
import math
from scipy.io import savemat
import time
import scipy.io


# 指定使用第 0 张 GPU 卡
os.environ["CUDA_VISIBLE_DEVICES"] = "8"

# 在代码中设置环境变量
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

# Attack switches
attack_enable_clean = False
attack_enable_gaussian = False
attack_enable_ours_f = False
attack_enable_ours = False
attack_enable_comparison = True
adveresarial_training_enable = False

# 创建日志文件夹
if log_enabled:
    log_folder = 'Defence Results My picture23'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# 配置日志
    log_filename = os.path.join(log_folder, f'AT_TR_compare_{current_time}.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

tl.set_backend('pytorch')  # Set tensorly to use PyTorch backend

class OptimizationConfig:
    def __init__(self, epsilon, lr, outer_num_steps, inner_num_steps, outer_dis_num, inner_dis_num, inner_tol):
        self.epsilon = epsilon
        self.lr = lr
        self.outer_num_steps = outer_num_steps
        self.inner_num_steps = inner_num_steps
        self.outer_dis_num = outer_dis_num
        self.inner_dis_num = inner_dis_num
        self.inner_tol = inner_tol

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


def objective_function_ATTR(X, G_list, E, inner_num_steps = 100, inner_tol = 1e-10):
    """
    Compute the objective function ||X - TN([G^(k)])||_F^2.
    X: Original tensor (PyTorch tensor)
    G_list: List of factor tensors in tensor train format
    E: Perturbation tensor (PyTorch tensor)
    """

    X_tube = X + E

    tr_cores = tr.tr_decompose2(X_tube, G_list, max_iter=inner_num_steps, tol=inner_tol, dis_num = 2000, device="cuda")

    TN_G = tl.tr_to_tensor(tr_cores)  # Reconstruct tensor from tensor train

    return -torch.norm(X - TN_G) ** 2, TN_G, tr_cores

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
    )

    TN_G = tl.tr_to_tensor(tr_cores)  # Reconstruct tensor from tensor train

    return -torch.norm(X - TN_G) ** 2, TN_G, tr_cores

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




def optimize_perturbation_simplifed(X, G_list_ori, config: OptimizationConfig):
    rse_E_list = []
    loss_list = []

    # Initialize perturbation tensor \mathcal{E}
    E = torch.randn_like(X, device='cuda')

    # 计算每个维度的权重
    total_size = sum(X.size())
    weights = [X.size(dim) / total_size for dim in range(X.dim())]

    
    # Apply the constraint ||E||_F^2 <= epsilon
    scaling_factor = (config.epsilon ** 0.5) / torch.norm(E)
    if torch.norm(E)**2 > config.epsilon:
        E.data = (scaling_factor) * E

    for step in range(config.outer_num_steps):
        # with torch.no_grad():
        G_list = tr.tr_decompose2(X+E, G_list_ori, max_iter=inner_num_steps, tol=inner_tol, dis_num=config.inner_dis_num, device="cuda")
        TN_G = tl.tr_to_tensor(G_list)  # Reconstruct tensor from tensor train

        E_old = E.clone().detach()

        # 初始化 E 的梯度为零
        E_sum = torch.zeros_like(E)
        # 遍历每个维度，计算梯度并累加
        for n in range(X.dim()):
            X_dims= list(range(X.dim()))
            new_X_dims = [X_dims[n]] + X_dims[n+1:] + X_dims[:n]
            X_unfold = tr.generalized_k_unfolding(X, new_X_dims, 0)

            # 计算除 cores[n] 之外的张量积
            X_hat = tr.contract_except_n(G_list, n)
            dims = list(range(X_hat.dim()))
            new_dims = [dims[-1]] + dims[:-1]
            Gr_unfold = tr.generalized_k_unfolding(X_hat, new_dims, 1)

            config.lr = 0.1
            E_unfold = tr.generalized_k_unfolding(E, new_X_dims, 0)
            E_unfold_grad = (X_unfold - (X_unfold + E_unfold) @ torch.linalg.pinv(Gr_unfold) @ Gr_unfold) @ Gr_unfold.T @ torch.linalg.pinv(Gr_unfold).T
            E_unfold = E_unfold - config.lr * E_unfold_grad

            E_folded = E_unfold.reshape(*[X.shape[dim] for dim in new_X_dims])
            E = E_folded.permute(*[new_X_dims.index(i) for i in range(len(new_X_dims))])
            E_sum += weights[n] * E
        # 更新 E
        E = E_sum
        
        scaling_factor = (config.epsilon ** 0.5) / torch.norm(E)
        if torch.norm(E)**2 > config.epsilon:
            # E.data = config.epsilon * E / torch.norm(E)
            E.data = (scaling_factor) * E

        rse_E = torch.norm(E - E_old)
        rse_E = rse_E.cpu().detach().numpy()

        loss_X_TNG = torch.norm(X - TN_G)
        loss_X_TNG = loss_X_TNG.cpu().detach().numpy()
        loss_list.append(loss_X_TNG)

        norm_E = torch.norm(E)

        logging.info(f"Step {step}, RSE_E: {rse_E}, Loss_X_Ours: {loss_X_TNG}, normE: {norm_E}")

        rse_E_list.append(rse_E)
        if step % config.outer_dis_num == 0:
            print(f"Step {step}, RSE_E: {rse_E}, Loss: {loss_X_TNG}, normE: {norm_E}")
        
        E = E.detach()
        TN_G = TN_G.detach()
        G_list = [G.detach() for G in G_list]
        
       

    return E, TN_G, loss_list

def optimize_perturbation(X, G_list_ori, config: OptimizationConfig):
    """
    Optimize perturbation tensor \mathcal{E} using gradient descent.
    X: Original tensor (PyTorch tensor)
    G_list: List of factor tensors in tensor train format
    epsilon: Frobenius norm constraint for \mathcal{E}
    lr: Learning rate for gradient descent
    num_steps: Number of gradient descent steps
    """
    # X = X.detach()
    loss_list = []
    # Initialize perturbation tensor \mathcal{E}
    E = torch.randn_like(X, requires_grad=True, device='cuda')
    # Apply the constraint ||E||_F^2 <= epsilon
    scaling_factor = (config.epsilon ** 0.5) / torch.norm(E)
    if torch.norm(E) ** 2 > config.epsilon:
        E.data = scaling_factor * E

    loss_values = []

    optimizer = torch.optim.Adam([E], lr=config.lr)

    for step in range(config.outer_num_steps):
        optimizer.zero_grad() # Clear gradients

        E_old = E.clone().detach()

        X.requires_grad_(True)
            
        # Compute loss
        loss, TN_G, G_list = objective_function_ATTR(X, G_list_ori, E, config.inner_num_steps, config.inner_tol)

        # print(f"TN_G max: {TN_G.max()}, min: {TN_G.min()}")

        # 检查损失是否为 NaN
        if torch.isnan(loss):
            raise ValueError("Loss is NaN")
    
        # grads = torch.autograd.grad(loss, E, retain_graph=True, allow_unused=True)
        # print(f"Gradient of E: {grads}")

        # # 梯度裁剪
        # torch.nn.utils.clip_grad_norm_([E], max_norm=0.1)

        # 在反向传播之前检查 E 的梯度
        if E.grad is not None:
            print("反向传播之前的梯度：", E.grad)
        # Backpropagation
        loss.backward()
        # print(loss.requires_grad)  # 确保损失函数需要梯度

        # 检查梯度
        # print("梯度在反向传播后的值：", E.grad)
        print(f"Gradient max: {E.grad.max()}, min: {E.grad.min()}")
        if torch.isnan(E.grad).any():
            break

        # Update E
        optimizer.step()

        # print(f"E max: {E.max()}, min: {E.min()}")

        # E_plus_X = X + E
        # E_plus_X_clamp = torch.clamp(E_plus_X, 0, 1)
        # E.data = E_plus_X_clamp - X

        # Apply the constraint ||E||_F^2 <= epsilon
        scaling_factor = (config.epsilon ** 0.5) / torch.norm(E)
        if torch.norm(E) ** 2 > config.epsilon:
            E.data = scaling_factor * E

        # 检查 E 是否为 NaN
        if torch.isnan(E).any():
            raise ValueError("E contains NaN after update")
        
        rse_E = torch.norm(E - E_old)

        loss_X_TNG = torch.norm(X - TN_G)
        loss_X_TNG = loss_X_TNG.cpu().detach().numpy()
        loss_list.append(loss_X_TNG)

        norm_E = torch.norm(E)

        # 记录 norm2 结果到日志
        loss_sqrt = math.sqrt(abs(loss))
        loss_values.append(loss_sqrt)
        logging.info(f"Step {step}, RSE_E: {rse_E}, Loss: {loss_X_TNG}, normE: {norm_E}")

        if step % config.outer_dis_num == 0:
            print(f"Step {step}, RSE_E: {rse_E}, Loss: {loss_X_TNG}, normE: {norm_E}")

        # TN_G2 = TN_G.clone().detach()

        
            

    return E, TN_G, loss_list

def optimize_perturbation_comparison(X, G_list, config: OptimizationConfig):
    """
    Optimize perturbation tensor \mathcal{E} using gradient descent for comparison.
    X: Original tensor (PyTorch tensor)
    G_list: List of factor tensors in tensor train format
    config: Optimization configuration
    """
    # Initialize perturbation tensor \mathcal{E}
    E = torch.randn_like(X, requires_grad=True, device='cuda')
    X = X.detach()
    # X.requires_grad_(True)
    G_list = [G.detach() for G in G_list]
    # Apply the constraint ||E||_F^2 <= epsilon
    scaling_factor = (config.epsilon ** 0.5) / torch.norm(E)
    if torch.norm(E)**2 > config.epsilon:
        E.data = scaling_factor * E

    optimizer = torch.optim.Adam([E], lr=config.lr)

    rec_loss_values = []

    tr_cores = [G.detach() for G in G_list]
    for step in range(config.outer_num_steps):
        optimizer.zero_grad()

        # Compute loss
        loss2, TN_G, tr_cores = objective_function_ATR(X, tr_cores, E, config.inner_num_steps, config.inner_tol)

        E_old = E.clone().detach()

        # Backpropagation
        loss2.backward()

        # Update E
        optimizer.step()

        # Apply the constraint ||E||_F^2 <= config.epsilon
        scaling_factor = (config.epsilon ** 0.5) / torch.norm(E)
        if torch.norm(E)**2 > config.epsilon:
            E.data = scaling_factor * E

        rse_E = torch.norm(E - E_old)
        
        norm2_X_TN_G = torch.norm(X - TN_G).item()
        rec_loss_values.append(norm2_X_TN_G)

        norm_E = torch.norm(E)
        logging.info(f"Comparison Step {step}, Loss_X_TNG: {norm2_X_TN_G}, normE: {norm_E}")

        if step != 0 and step % config.outer_dis_num == 0:
            print(f"Comparison Step {step}, Loss: {norm2_X_TN_G}, normE: {norm_E}")


            

    # return E, TN_G, rec_loss_values
    return E_old, TN_G, rec_loss_values

def optimize_G_list(X, G_list_ori, E, config: OptimizationConfig):
    """
    Optimize the tensor train cores G_list_ori using gradient descent.
    X: Original tensor (PyTorch tensor)
    G_list_ori: List of factor tensors in tensor train format (PyTorch tensors)
    config: Optimization configuration
    """
    # 确保 G_list_ori 的每个张量启用梯度
    G_list = [G.clone().detach().requires_grad_(True) for G in G_list_ori]

    # 定义优化器，只优化 G_list
    optimizer = torch.optim.Adam(G_list, lr=config.lr)

    loss_list = []

    for step in range(config.outer_num_steps):
        optimizer.zero_grad()  # 清除梯度

        # 计算损失（负数）
        loss, TN_G, _ = objective_function_DFTR(X, G_list, E, inner_num_steps=config.inner_num_steps, inner_tol=config.inner_tol)
        loss = -loss  # 最小化负数的损失

        # 反向传播
        loss.backward()
        TN_G_det = TN_G.detach()
        G_list_det = [core.detach() for core in G_list]

        # 更新 G_list
        optimizer.step()

        # 记录损失
        loss_value = loss.item()
        loss_list.append(loss_value)

        loss_X_TNG = torch.norm(X - TN_G)
        loss_X_TNG = loss_X_TNG.cpu().detach().numpy()

        # 打印日志
        if step % config.outer_dis_num == 0:
            print(f"Step {step}, Loss: {loss_X_TNG}")
            logging.info(f"Step {step}, Loss: {loss_X_TNG}")

        # 检查是否有 NaN
        if torch.isnan(loss):
            raise ValueError("Loss is NaN")

    # 返回优化后的 G_list 和损失列表
    return G_list, loss_list



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
):
    if num_rounds <= 0:
        raise ValueError('num_rounds must be a positive integer')

    original_attack_steps = attack_config.outer_num_steps
    original_defence_steps = defence_config.outer_num_steps
    attack_config.outer_num_steps = 1
    defence_config.outer_num_steps = 1

    current_G = _clone_cores(G_list_init)
    best_state = _clone_cores(current_G)
    best_loss = float('inf')
    history = []
    attack_E = None

    try:
        for round_idx in range(num_rounds):
            attack_E, _, attack_losses = optimize_perturbation(
                X,
                _clone_cores(current_G),
                attack_config,
            )

            optimized_G_list, defence_losses = optimize_G_list(
                X,
                _clone_cores(current_G),
                attack_E,
                defence_config,
            )

            current_G = _clone_cores(optimized_G_list)
            loss_last = float(defence_losses[-1]) if defence_losses else float('inf')

            if loss_last < best_loss:
                best_loss = loss_last
                best_state = _clone_cores(current_G)

            if track_history:
                history_entry = {
                    'round': round_idx + 1,
                    'attack_loss_curve': [float(val) for val in attack_losses],
                    'defence_loss_curve': [float(val) for val in defence_losses],
                    'attack_loss_final': float(attack_losses[-1]) if attack_losses else None,
                    'defence_loss_final': loss_last,
                }
                if store_perturbations and attack_E is not None:
                    history_entry['perturbation'] = attack_E.detach().cpu()
                history.append(history_entry)
    finally:
        attack_config.outer_num_steps = original_attack_steps
        defence_config.outer_num_steps = original_defence_steps

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

    # Decompose X into tensor train format
    ranks = [5, 5, 5]  # Rank for each factor tensor
    # ranks = [7, 7, 7]  # Rank for each factor tensor
    ranks = ranks + [ranks[0]]  # 最后一个 rank 应该等于第一个 rank
    shape = list(X.shape)
    tr_cores = tr.initialize_tr_cores(shape, ranks, device="cuda")
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

    if attack_enable_ours_f:
        start_time = time.time()
        E_optimized_F, TN_G_F, loss_ours_F = optimize_perturbation_simplifed(X, tr_cores, config)
        time_simplified = time.time() - start_time
        print(f"optimize_perturbation_simplifed 运行时间: {time_simplified:.4f} 秒")
        with torch.no_grad():
            norm_attack_ours_F = torch.norm(X - TN_G_F).item()
    else:
        print("optimize_perturbation_simplifed 已跳过 (attack_enable_ours_f=False)")

    E_optimized = None
    TN_G = None
    loss_ours = []
    time_optimized = None
    norm_attack_ours = None

    if attack_enable_ours:
        X.requires_grad_(True)
        start_time = time.time()
        E_optimized, TN_G, loss_ours = optimize_perturbation(X, tr_cores, config)
        time_optimized = time.time() - start_time
        print(f"optimize_perturbation 运行时间: {time_optimized:.4f} 秒")
        with torch.no_grad():
            norm_attack_ours = torch.norm(X - TN_G).item()
    else:
        print("optimize_perturbation 已跳过 (attack_enable_ours=False)")

    X = X.detach()
    tr_cores = [G.detach() for G in tr_cores]

    norm2_original_TR_algorithm = None
    zero_perturbation = None
    if attack_enable_clean:
        tr_cores2 = tr.tr_decompose2(X, tr_cores, max_iter=config.inner_num_steps, tol=inner_tol, device="cuda")
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
        E_Gaus = torch.normal(mean=0, std=1, size=X.shape, device="cuda")
        scaling_factor = (config.epsilon ** 0.5) / torch.norm(E_Gaus)
        if torch.norm(E_Gaus)**2 > config.epsilon:
            E_Gaus.data = scaling_factor * E_Gaus
        tr_cores3 = tr.tr_decompose2(X + E_Gaus, tr_cores, max_iter=config.inner_num_steps, tol=inner_tol, device="cuda")
        TN_G_Gaus = tl.tr_to_tensor(tr_cores3)
        norm2_Gaus = torch.norm(X - TN_G_Gaus).item()
        print(f"Norm2 (Gaussian Noise): {norm2_Gaus}")
        logging.info(f"Norm2 (Gaussian Noise): {norm2_Gaus}")

    E_comparison = None
    norm2_comparison_algorithm = None
    if attack_enable_comparison:
        start_time = time.time()
        E_comparison, TN_G_comparison, loss_ATR = optimize_perturbation_comparison(X, tr_cores, config)
        time_comp = time.time() - start_time
        print(f"time_comp 运行时间: {time_comp:.4f} 秒")
        norm2_comparison_algorithm = torch.norm(X - TN_G_comparison).item()
        print(f"Norm2 (Comparison Algorithm): {norm2_comparison_algorithm}")
        logging.info(f"Norm2 (Comparison Algorithm): {norm2_comparison_algorithm}")
        E_comparison_cpu = E_comparison.cpu().detach().numpy()
        X_E_comparison_cpu = (X + E_comparison).cpu().detach().numpy()
        TN_G_comparison = (TN_G_comparison).cpu().detach().numpy()

    perturbations_for_defence = []

    if attack_enable_clean and zero_perturbation is not None and norm2_original_TR_algorithm is not None:
        perturbations_for_defence.append(("Clean", zero_perturbation, norm2_original_TR_algorithm))

    if attack_enable_gaussian and E_Gaus is not None and norm2_Gaus is not None:
        perturbations_for_defence.append(("Gaussian", E_Gaus, norm2_Gaus))

    if attack_enable_ours_f and E_optimized_F is not None and norm_attack_ours_F is not None:
        perturbations_for_defence.append(("Ours_F", E_optimized_F, norm_attack_ours_F))

    if attack_enable_ours and E_optimized is not None and norm_attack_ours is not None:
        perturbations_for_defence.append(("Ours", E_optimized, norm_attack_ours))

    if attack_enable_comparison and E_comparison is not None and norm2_comparison_algorithm is not None:
        perturbations_for_defence.append(("Comparison", E_comparison, norm2_comparison_algorithm))

    results_table = []

    if not perturbations_for_defence:
        print('未启用任何攻击开关，跳过 Defence 评估。')

    for label, perturbation, attack_norm in perturbations_for_defence:
        optimized_G_list, loss_list = optimize_G_list(X, tr_cores, perturbation, config)

        # 重新初始化一份 G_list，用作 TR-ALS 的起点
        G_list_init = [core.clone().detach() for core in optimized_G_list]

        with torch.no_grad():
            eval_loss, TN_G_eval, _ = objective_function_DFTR(
                X,
                G_list_init,
                perturbation,
                inner_num_steps=config.inner_num_steps,
                inner_tol=config.inner_tol,
            )

        defence_norm = torch.norm(X - TN_G_eval).item()
        results_table.append({"label": label, "attack_norm": attack_norm, "defence_norm": defence_norm})

        print(f"Norm (Defence TR Algorithm - {label}): {defence_norm}")
        logging.info(f"Norm (Defence TR Algorithm - {label}): {defence_norm}")

    if results_table:
        header = f"{'Case':<12}{'Attack Norm':>15}{'Defence Norm':>18}"
        print(header)
        print('-' * len(header))
        for row in results_table:
            print(f"{row['label']:<12}{row['attack_norm']:>15.6f}{row['defence_norm']:>18.6f}")
            logging.info(
                f"Table Row -> Case: {row['label']}, Attack Norm: {row['attack_norm']}, Defence Norm: {row['defence_norm']}"
            )

    if adveresarial_training_enable:
        print('\n=== Adversarial Comparison Summary ===')
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

    if attack_enable_comparison and imshow_enabled and imshow_requirements_met:
        
        # 显示 TN_G 图像
        plt.imshow(X_E_comparison_cpu)
        save_mat_file(output_folder, f'X_E_comparison_{base_name}', X_E_comparison_cpu)
        plt.axis('off')

        # 保存图像
        fig = plt.gcf()  # 获取当前图形对象
        save_fig(output_folder, f'X_E_comparison_{base_name}', fig)  # 保存图形

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
        plt.plot(loss_ATR, label='Comparison')
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
        plt.plot(loss_ATR, label='Comparison', color='green')
        plt.xlabel('Step')
        plt.ylabel('Loss (sqrt)')
        plt.title('Loss over Steps (Comparison)')
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

    return time_simplified, time_optimized


def process_image(image_path, output_folder, config):
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    X = torch.tensor(image / 255.0, dtype=torch.float32)
    base_name = os.path.basename(image_path).split('.')[0]
    return process_tensor_data(X, base_name, output_folder, config)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial Tensor Ring experiments on images or synthetic tensors.")
    parser.add_argument(
        "--data-source",
        choices=["image", "synthetic"],
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
        default=(5, 5, 5),
        help="Tensor Ring ranks for each mode of the synthetic tensor.",
    )
    parser.add_argument(
        "--synthetic-seed",
        type=int,
        default=None,
        help="Optional random seed for synthetic tensor generation.",
    )
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = os.path.join(log_folder, 'output')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    epsilon = 100
    logging.info(f"Norm2 (epsilon): {epsilon}")
    inner_tol = 1e-8
    lr = 0.01
    outer_num_steps = 100
    inner_num_steps = 100
    outer_dis_num = 1
    inner_dis_num = 2000
    logging.info("Optimization parameters:")
    logging.info(f"epsilon (Frobenius norm constraint): {epsilon}")
    logging.info(f"inner_tol (Tolerance for inner optimization): {inner_tol}")
    logging.info(f"lr (Learning rate): {lr}")
    logging.info(f"outer_num_steps (Number of outer optimization steps): {outer_num_steps}")
    logging.info(f"inner_num_steps (Number of inner optimization steps): {inner_num_steps}")
    logging.info(f"outer_dis_num (Display loss every outer_dis_num steps): {outer_dis_num}")
    logging.info(f"inner_dis_num (Display loss every inner_dis_num steps): {inner_dis_num}")

    config = OptimizationConfig(epsilon, lr, outer_num_steps, inner_num_steps, outer_dis_num, inner_dis_num, inner_tol)
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

            time_simplified, time_optimized = process_image(image_path, output_folder, config)
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
