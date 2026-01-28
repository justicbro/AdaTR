import torch
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
import pandas as pd
from skimage.metrics import structural_similarity as ssim


# 指定使用第 0 张 GPU 卡
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

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
compare_method_enabled = False
save_mat_enabled = False
save_fig_enabled = False

# 创建日志文件夹
if log_enabled:
    log_folder = 'Resluts My picture24'
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

def _safe_to_float(x):
    try:
        return float(x)
    except Exception:
        # 如果是标量张量/ndarray，取 item
        try:
            return float(np.asarray(x).item())
        except Exception:
            return np.nan

def run_one_case(name, input_tensor, **kwargs):
    """
    统一调用 optimize_perturbation_comparison，并把常见的输出指标取出来。
    返回 dict，键里尽量兼容你代码里常用的命名：
      - loss_ours_last, loss_ours_F_last, loss_ATR_last, loss_TR_last 或 baseline
      - 也会把 'final_loss' / 'loss' / 'recon_error' 之类兜底收集
    """
    tr_cores = kwargs.get('tr_cores', None)
    config = kwargs.get('config', None)
    out = optimize_perturbation_comparison(input_tensor, tr_cores, config)

    # 兼容不同返回风格：dict / tuple / obj
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
        result = out[1]
    elif isinstance(out, dict):
        result = out
    else:
        # 如果是对象，尝试取 __dict__
        result = getattr(out, "__dict__", {}) if not isinstance(out, dict) else out

    # 可能的键名列表（按优先级）
    keys_map = {
        "ours":    ["loss_ours_last", "loss_ours", "ours_last", "ada_loss", "ada_final"],
        "ours_F":  ["loss_ours_F_last", "loss_ours_F", "ours_F_last", "fag_loss", "fag_final"],
        "ATTR":    ["loss_ATR_last", "loss_attr_last", "attr_last", "attr_loss", "attr_final"],
        "TR":      ["loss_TR_last", "loss_tr_last", "tr_last", "baseline", "clean", "tr_final"],
        "final":   ["final_loss", "loss", "recon_error", "reconstruction_error"]
    }

    parsed = {}
    for k_alias, candidates in keys_map.items():
        for cand in candidates:
            if cand in result:
                parsed[k_alias] = _safe_to_float(result[cand])
                break
        # 没取到也写个 NaN 占位
        parsed.setdefault(k_alias, np.nan)

    parsed["case"] = name
    return parsed

def batch_compare_runtime_and_losses(X, TN_G_ori, TN_G_Gaus, E_optimized, E_optimized_F, **kwargs):
    """
    四种输入统一评测：
      1) TR factors (clean)：TN_G_ori
      2) TR factors (Gaussian)：TN_G_Gaus
      3) X + E_optimized
      4) X + E_optimized_F
    说明：
      - 如果 optimize_perturbation_comparison 期望输入是“观测张量”而不是因子，
        那么把 TN_G_* 换成它们对应的重构张量（例如 TR_compose(TN_G_ori)）。
      - 如果期望输入是因子，X+E_* 两项请改成把 X+E 先分解（或你已有的接口）。
    """
    # 从 kwargs 中提取额外参数
    tr_cores = kwargs.get('tr_cores', None)
    config = kwargs.get('config', None)
    
    inputs = [
        ("TR (clean factors)",  TN_G_ori),
        ("TR (gaussian factors)", TN_G_Gaus),
        ("X + E_optimized",     X + E_optimized),
        ("X + E_optimized_F",   X + E_optimized_F),
    ]

    rows = []
    for name, tensor_in in inputs:
        rows.append(run_one_case(name, tensor_in, **kwargs))

    df = pd.DataFrame(rows, columns=["case", "ours", "ours_F", "ATTR", "TR", "final"])
    # 打印一个整洁表
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print("\n=== Comparison on four inputs ===")
        print(df.to_string(index=False))

    # 可选：保存 CSV
    try:
        df.to_csv("comparison_four_inputs.csv", index=False)
        print("\nSaved results to comparison_four_inputs.csv")
    except Exception as e:
        print(f"[warn] failed to save csv: {e}")

    return df




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


def compute_normalized_rse(reference, reconstruction):
    """Return ||reference - reconstruction||_F / ||reference||_F; NaN if denominator vanishes."""
    diff_norm = torch.norm(reference - reconstruction)
    ref_norm = torch.norm(reference)
    if ref_norm.item() == 0:
        return float('nan')
    return (diff_norm / ref_norm).item()

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

    # 调整布局以避免标题重叠

    if save_path:
        plt.savefig(save_path)
    plt.show()

def objective_function_ATTR(X, G_list, E, inner_num_steps = 100, inner_tol = 1e-10):
    """
    Compute the objective function ||X - TN([G^(k)])||_F^2.
    X: Original tensor (PyTorch tensor)
    G_list: List of factor tensors in tensor train format
    E: Perturbation tensor (PyTorch tensor)
    """

    X_tube = X + E

    tr_cores = tr.tr_decompose2(X_tube, G_list, max_iter=inner_num_steps, tol=inner_tol, dis_num = 2000, device="cuda")


    # 确保 tr_cores 启用了梯度追踪

    TN_G = tl.tr_to_tensor(tr_cores)  # Reconstruct tensor from tensor train
    # 在计算损失之前检查 TN_G 的状态
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

def objective_function_ATTR_simp(X, G_list, E, E_unfold, inner_num_steps, inner_tol):
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
    G_list = tr.tr_decompose2(X+EC, G_list, max_iter=inner_num_steps, tol=inner_tol, dis_num = 2000, device="cuda")
    TN_G = tl.tr_to_tensor(G_list)  # Reconstruct tensor from tensor train

    n = X.dim() - 2
    X_dims= list(range(X.dim()))
    new_X_dims = [X_dims[n]] + X_dims[n+1:] + X_dims[:n]
    X_unfold = tr.generalized_k_unfolding(X, new_X_dims, 0)
    X_unfold = torch.tensor(X_unfold, requires_grad=True, dtype=torch.float32).to('cuda')

    # 计算除 cores[n] 之外的张量积
    X_hat = tr.contract_except_n(G_list, n)
    dims = list(range(X_hat.dim()))
    new_dims = [dims[-1]] + dims[:-1]
    Gr_unfold = tr.generalized_k_unfolding(X_hat, new_dims, 1)

    loss_values = []

    optimizer = torch.optim.Adam([E_unfold], lr=config.lr)
    return -(torch.norm(X_unfold - (X_unfold + E_unfold) @ torch.linalg.pinv(Gr_unfold) @ Gr_unfold) ** 2), TN_G  # Example: Minimize the norm instead of maximizing

def optimize_perturbation_simplified_2(X, G_list, config: OptimizationConfig):
    """
    Optimize perturbation tensor \mathcal{E} using gradient descent for comparison.
    X: Original tensor (PyTorch tensor)
    G_list: List of factor tensors in tensor train format
    config: Optimization configuration
    """
    # Initialize perturbation tensor \mathcal{E}
    E = torch.randn_like(X, requires_grad=True, device='cuda')
    # Apply the constraint ||E||_F^2 <= epsilon
    scaling_factor = (config.epsilon ** 0.5) / torch.norm(E)
    if torch.norm(E)**2 > config.epsilon:
        E.data = scaling_factor * E

    # Compute loss
    n = X.dim() - 2
    X_dims= list(range(X.dim()))
    new_X_dims = [X_dims[n]] + X_dims[n+1:] + X_dims[:n]
    E_unfold = tr.generalized_k_unfolding(E, new_X_dims, 0)
    E_unfold = torch.tensor(E_unfold, requires_grad=True, dtype=torch.float32).to('cuda')

    optimizer = torch.optim.Adam([E_unfold], lr=config.lr)

    rec_loss_values = []
    norm2_X_TN_G_list = []

    for step in range(config.outer_num_steps):
        optimizer.zero_grad()

        # Compute loss
        n = X.dim() - 2
        X_dims= list(range(X.dim()))
        new_X_dims = [X_dims[n]] + X_dims[n+1:] + X_dims[:n]
        E_unfold = tr.generalized_k_unfolding(E, new_X_dims, 0)
        E_unfold = torch.tensor(E_unfold, requires_grad=True, dtype=torch.float32).to('cuda')
        loss, TN_G = objective_function_ATTR_simp(X, G_list, E, E_unfold, config.inner_num_steps, config.inner_tol)

        E_old = E.clone().detach()

        # Backpropagation
        loss.backward()

        # Update E
        optimizer.step()

        E_fold = E_unfold.reshape(*[X.shape[dim] for dim in new_X_dims])
        E = E_fold.permute(*[new_X_dims.index(i) for i in range(len(new_X_dims))])

        # Apply the constraint ||E||_F^2 <= config.epsilon
        scaling_factor = (config.epsilon ** 0.5) / torch.norm(E)
        if torch.norm(E)**2 > config.epsilon:
            E.data = scaling_factor * E


        norm2_X_TN_G = torch.norm(X - TN_G).item()
        rec_loss_values.append(norm2_X_TN_G)
        norm2_X_TN_G_list.append(norm2_X_TN_G)

        norm_E = torch.norm(E)
        logging.info(f"Comparison Step {step}, Loss_X_TNG: {norm2_X_TN_G}, normE: {norm_E}, Loss: {loss.item()}")

        if step != 0 and step % config.outer_dis_num == 0:
            print(f"Comparison Step {step}, Loss: {norm2_X_TN_G}, normE: {norm_E}, Loss: {loss.item()}")
            

    return E, TN_G, norm2_X_TN_G_list



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


        # 检查损失是否为 NaN
        if torch.isnan(loss):
            raise ValueError("Loss is NaN")
    

        # 梯度裁剪

        # 在反向传播之前检查 E 的梯度
        if E.grad is not None:
            print("反向传播之前的梯度：", E.grad)
        # Backpropagation
        loss.backward()

        # 检查梯度
        print(f"Gradient max: {E.grad.max()}, min: {E.grad.min()}")
        if torch.isnan(E.grad).any():
            break

        # Update E
        optimizer.step()



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

        # 检查梯度

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

            

    return E, TN_G, rec_loss_values

def optimize_perturbation_comparison_2(X, G_list, config: OptimizationConfig):
    """
    Optimize perturbation tensor \mathcal{E} using gradient descent for comparison.
    X: Original tensor (PyTorch tensor)
    G_list: List of factor tensors in tensor train format
    config: Optimization configuration
    """
    # Initialize perturbation tensor \mathcal{E}
    E = torch.randn_like(X, device='cuda')
    E = E.detach()
    X = X.detach()
    G_list = [G.detach() for G in G_list]
    # Apply the constraint ||E||_F^2 <= epsilon
    scaling_factor = (config.epsilon ** 0.5) / torch.norm(E)
    if torch.norm(E)**2 > config.epsilon:
        E.data = scaling_factor * E

    optimizer = torch.optim.Adam([E], lr=config.lr)

    rec_loss_values = []

    for step in range(config.outer_num_steps):
        optimizer.zero_grad()

        # Compute loss
        EC = E.clone().detach()
        tr_cores = tr.tr_decompose2(X+EC, G_list, max_iter=inner_num_steps, tol=inner_tol, dis_num = 2000, device="cuda")
        TN_G = tl.tr_to_tensor(tr_cores)  # Reconstruct tensor from tensor train
        E = E - config.lr * 2 * (X + E - TN_G)

        E_old = E.clone().detach()


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

            

    return E, TN_G, rec_loss_values

def process_image(image_path, output_folder, config):
    # 读取图片
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    X = torch.tensor(image / 255.0, dtype=torch.float32).to("cuda")

    # Decompose X into tensor train format
    ranks = [5, 5, 5]  # Rank for each factor tensor
    ranks = ranks + [ranks[0]]  # 最后一个 rank 应该等于第一个 rank
    shape = list(X.shape)
    tr_cores = tr.initialize_tr_cores(shape, ranks, device="cuda")
    # 将 tr_cores 转换为 NumPy 数组
    tr_cores_np = [core.cpu().numpy() for core in tr_cores]

     # 保存结果
    base_name = os.path.basename(image_path).split('.')[0]
    # 创建一个字典来存储所有的核心张量
    tr_cores_dict = {f'tr_core_{i}': tr_cores_np[i] for i in range(len(tr_cores_np))}

    if save_mat_enabled:
        # 保存字典到 .mat 文件
        scipy.io.savemat(os.path.join(output_folder, f'tr_cores_{base_name}.mat'), tr_cores_dict)

    # Optimize perturbation \mathcal{E}
    start_time = time.time()
    E_optimized_F, TN_G_F, loss_ours_F = optimize_perturbation_simplifed(X, tr_cores, config)
    time_simplified = time.time() - start_time

    X.requires_grad_(True)
    start_time = time.time()
    E_optimized, TN_G, loss_ours = optimize_perturbation(X, tr_cores, config)
    time_optimized = time.time() - start_time

    print(f"optimize_perturbation_simplifed 运行时间: {time_simplified:.4f} 秒")
    print(f"optimize_perturbation 运行时间: {time_optimized:.4f} 秒")

    X = X.detach()
    tr_cores = [G.detach() for G in tr_cores]

    E_Gaus = torch.normal(mean=0, std=1, size=X.shape, device="cuda")
    # Apply the constraint ||E||_F^2 <= epsilon
    scaling_factor = (config.epsilon ** 0.5) / torch.norm(E_Gaus)
    if torch.norm(E_Gaus)**2 > config.epsilon:
        E_Gaus.data = scaling_factor * E_Gaus



    # 定义输入列表
    inputs = [
        ("X", X),
        ("X+Gaus", X+E_Gaus),
        ("X + E_optimized", X + E_optimized),
        ("X + E_optimized_F", X + E_optimized_F),
    ]

    # 初始化结果矩阵
    results = []

    # 遍历每个输入，调用 optimize_perturbation_comparison 和 tr_decompose2 并计算指标
    for name, input_tensor in inputs:
        # 调用 optimize_perturbation_comparison
        tempepsilon = config.epsilon
        config.epsilon = 10
        E_comparison, TN_G_comparison, _ = optimize_perturbation_comparison(input_tensor, tr_cores, config)

        # 调用 tr_decompose2
        tr_cores_decomposed = tr.tr_decompose2(input_tensor, tr_cores, max_iter=inner_num_steps, tol=config.inner_tol, device="cuda")
        TN_G_decomposed = tl.tr_to_tensor(tr_cores_decomposed)

        # 计算 PSNR
        psnr_comparison = calculate_psnr(X.cpu().numpy(), TN_G_comparison.cpu().detach().numpy())
        psnr_decomposed = calculate_psnr(X.cpu().numpy(), TN_G_decomposed.cpu().detach().numpy())

        # 计算 RSE
        rse_comparison = compute_normalized_rse(X, TN_G_comparison)
        rse_decomposed = compute_normalized_rse(X, TN_G_decomposed)

        # 获取图像的最小维度
        min_dim = min(X.shape[0], X.shape[1])  # 假设 X 的形状为 (H, W, C)

        # 设置窗口大小为小于或等于最小维度的最大奇数
        win_size = min(7, min_dim) if min_dim >= 7 else (min_dim if min_dim % 2 == 1 else min_dim - 1)

        # 检查图像尺寸是否小于 7x7
        if min_dim < 7:
            print(f"Skipping SSIM calculation for {name} due to small image size.")
            ssim_comparison = None
            ssim_decomposed = None
        else:
            # 计算 SSIM
            ssim_comparison = ssim(
                X.cpu().numpy(),
                TN_G_comparison.cpu().detach().numpy(),
                multichannel=True,
                win_size=win_size,
                channel_axis=-1,
                data_range=1.0
            )
            ssim_decomposed = ssim(
                X.cpu().numpy(),
                TN_G_decomposed.cpu().detach().numpy(),
                multichannel=True,
                win_size=win_size,
                channel_axis=-1,
                data_range=1.0
            )

        # 保存结果
        results.append([name, "optimize_perturbation_comparison", psnr_comparison, rse_comparison, ssim_comparison])
        results.append([name, "tr_decompose2", psnr_decomposed, rse_decomposed, ssim_decomposed])

     # 转换为 DataFrame 并保存
    df_results = pd.DataFrame(results, columns=["Input", "Method", "PSNR", "RSE", "SSIM"])
    print(df_results)

    # 保存到 CSV 文件
    image_name = os.path.basename(image_path).split('.')[0]
    df_results.to_csv(os.path.join(output_folder, f"{image_name}_results.csv"), index=False)
    print(f"Results saved to {os.path.join(output_folder, f'{image_name}_results.csv')}")



    if compare_method_enabled:
        # Optimize perturbation \mathcal{E} using comparison algorithm
        start_time = time.time()
        E_comparison, TN_G_comparison, loss_ATR =  optimize_perturbation_comparison(X, tr_cores, config)
        time_comp = time.time() - start_time
        print(f"time_comp 运行时间: {time_comp:.4f} 秒")
        norm2_comparison_algorithm = torch.norm(X - TN_G_comparison).item()
        print(f"Norm2 (Comparison Algorithm): {norm2_comparison_algorithm}")
        logging.info(f"Norm2 (Comparison Algorithm): {norm2_comparison_algorithm}")
        E_comparison_cpu = E_comparison.cpu().detach().numpy()
        X_E_comparison_cpu = (X + E_comparison).cpu().detach().numpy()
        TN_G_comparison = (TN_G_comparison).cpu().detach().numpy()

    if imshow_enabled:
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

        # 保存结果
        base_name = os.path.basename(image_path).split('.')[0]
        # 假设 output_folder 已经定义
        # 显示 TN_G 图像
        plt.clf()
        plt.imshow(X_E_cpu)
        save_mat_file(output_folder, f'X_E_Ours_{base_name}', X_E_cpu)

        # 保存为 JPEG 格式
        plt.axis('off')
        fig = plt.gcf()  # 获取当前图形对象
        save_fig(output_folder, f'X_E_Ours_{base_name}', fig)  # 保存图形

        # 保存图像os.path.join(output_folder, f'{base_name}_X_E.png')

        plt.imshow(X_E_cpu_F)
        save_mat_file(output_folder, f'X_E_Ours_F_{base_name}', X_E_cpu_F)
        plt.axis('off')
        fig = plt.gcf()  # 获取当前图形对象
        save_fig(output_folder, f'X_E_Ours_F_{base_name}', fig)  # 保存图形

        # 显示 TN_G 图像
        plt.imshow(X_cpu)
        save_mat_file(output_folder, f'X_{base_name}', X_cpu)
        plt.axis('off')

        # 保存图像
        fig = plt.gcf()  # 获取当前图形对象
        save_fig(output_folder, f'X_{base_name}', fig)  # 保存图形

        plt.clf()
        plt.imshow(X_E_cpu_Gaus)
        save_mat_file(output_folder, f'X_E_Gaus_{base_name}', X_E_cpu_Gaus)
        plt.axis('off')

        # 保存图像os.path.join(output_folder, f'{base_name}_X_E.png')
        fig = plt.gcf()  # 获取当前图形对象
        save_fig(output_folder, f'X_E_Gaus_{base_name}', fig)  # 保存图形

    # 获取当前时间并格式化为字符串
    save_path = os.path.join(output_folder, f'comparison_{base_name}.png')

    if compare_method_enabled:
        
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

    return time_simplified, time_optimized, df_results

# Example usage
if __name__ == "__main__":
    # Example tensor X (3D for simplicity)
    # 读取 Lena 图片
    input_folder = './AT_TD/My picture'
    output_folder = os.path.join(log_folder, 'output')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

 
    # Decompose X into tensor train format
    ranks = [5, 5, 5]  # Rank for each factor tensor
    ranks = ranks + [ranks[0]]  # 最后一个 rank 应该等于第一个 rank

    # Set input parameter
    # epsilon = (100)**2 # Frobenius norm constraint for perturbation tensor
    epsilon = 10
    logging.info(f"Norm2 (epsilon): {epsilon}")
    inner_tol = 1e-8 # Tolerance for inner optimization
    lr = 0.01  # Learning rate for gradient descent
    outer_num_steps = 100 # Number of outer optimization steps
    inner_num_steps = 100 # Number of inner optimization steps
    outer_dis_num = 1 # Display loss every outer_dis_num steps
    inner_dis_num = 2000 # Display loss every inner_dis_num steps
    logging.info(f"Optimization parameters:")
    logging.info(f"epsilon (Frobenius norm constraint): {epsilon}")
    logging.info(f"inner_tol (Tolerance for inner optimization): {inner_tol}")
    logging.info(f"lr (Learning rate): {lr}")
    logging.info(f"outer_num_steps (Number of outer optimization steps): {outer_num_steps}")
    logging.info(f"inner_num_steps (Number of inner optimization steps): {inner_num_steps}")
    logging.info(f"outer_dis_num (Display loss every outer_dis_num steps): {outer_dis_num}")
    logging.info(f"inner_dis_num (Display loss every inner_dis_num steps): {inner_dis_num}")

    config = OptimizationConfig(epsilon, lr, outer_num_steps, inner_num_steps, outer_dis_num, inner_dis_num, inner_tol)

    times = []
    norms = []
    times_simplified = []
    times_optimized = []
    all_metrics = {
        "X": {"PSNR": [], "RSE": [], "SSIM": []},
        "X+Gaus": {"PSNR": [], "RSE": [], "SSIM": []},
        "X + E_optimized": {"PSNR": [], "RSE": [], "SSIM": []},
        "X + E_optimized_F": {"PSNR": [], "RSE": [], "SSIM": []},
    }
    for i in range(1, 8):
        image_name = f'{i}.png'
        image_path = os.path.join(input_folder, image_name)
        config.epsilon = epsilon
        if os.path.isfile(image_path):
            time_simplified, time_optimized, df_results = process_image(image_path, output_folder, config)
            times_simplified.append(time_simplified)
            # 假设 `process_image` 返回 `time_optimized`
            times_optimized.append(time_optimized)
            # 提取 PSNR、RSE 和 SSIM 指标
            psnr_values = df_results["PSNR"].tolist()
            rse_values = df_results["RSE"].tolist()
            ssim_values = df_results["SSIM"].tolist()

            # 将每张图片的指标添加到对应的输入类型中
            for _, row in df_results.iterrows():
                input_type = row["Input"]
                all_metrics[input_type]["PSNR"].append(row["PSNR"])
                all_metrics[input_type]["RSE"].append(row["RSE"])
                all_metrics[input_type]["SSIM"].append(row["SSIM"])

        # 计算均值和方差
        mean_simplified = np.mean(times_simplified)
        std_simplified = np.std(times_simplified)
        mean_optimized = np.mean(times_optimized)
        std_optimized = np.std(times_optimized)

        print(f"Simplified Algorithm: Mean = {mean_simplified:.4f}, Std = {std_simplified:.4f}")
        print(f"Optimized Algorithm: Mean = {mean_optimized:.4f}, Std = {std_optimized:.4f}")

        # 计算每个输入类型的均值和方差
        summary_results = []
        for input_type, metrics in all_metrics.items():
            psnr_mean, psnr_std = np.mean(metrics["PSNR"]), np.std(metrics["PSNR"])
            rse_mean, rse_std = np.mean(metrics["RSE"]), np.std(metrics["RSE"])
            ssim_mean, ssim_std = np.mean(metrics["SSIM"]), np.std(metrics["SSIM"])

            summary_results.append({
                "Input": input_type,
                "PSNR_Mean": psnr_mean, "PSNR_Std": psnr_std,
                "RSE_Mean": rse_mean, "RSE_Std": rse_std,
                "SSIM_Mean": ssim_mean, "SSIM_Std": ssim_std
            })

        # 保存统计结果到 summary_results.csv
        df_summary = pd.DataFrame(summary_results)
        df_summary.to_csv(os.path.join(output_folder, "summary_results.csv"), index=False)
        print(f"Summary results saved to {os.path.join(output_folder, 'summary_results.csv')}")






