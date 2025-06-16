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
import seaborn as sns


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
log_enabled = False
imshow_enabled = False
compare_method_enabled = True

# 创建日志文件夹
if log_enabled:
    log_folder = 'Resluts My picture6'
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
    # # 添加行标题
    # for row in range(rows):
    #     axes[row, 0].set_ylabel(row_titles[row], rotation=0, size='large', labelpad=40, ha='right', va='center')
    
    # # 添加列标题
    # for col in range(cols):
    #     axes[0, col].set_title(col_titles[col], pad=20, size='large')
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
        # ax.set_title(f"{titles[i]}\nPSNR: {psnr:.2f} dB")
        ax.set_title(f"\nPSNR: {psnr:.2f} dB")
        ax.axis('off')

    # # 调整布局以避免标题重叠
    # fig.tight_layout(rect=[0.15, 0.1, 1, 0.95])  # 留出足够的空间用于行标题和列标题
    # plt.subplots_adjust(left=0.2, right=0.95, top=0.85, bottom=0.1, wspace=0.3, hspace=0.3)

    if save_path:
        plt.savefig(save_path)
    plt.show()

def objective_function_ATTR(X, G_list, E, iner_num_steps = 100, inner_tol = 1e-10):
    """
    Compute the objective function ||X - TN([G^(k)])||_F^2.
    X: Original tensor (PyTorch tensor)
    G_list: List of factor tensors in tensor train format
    E: Perturbation tensor (PyTorch tensor)
    """
    # ranks = []
    # for core in G_list:
    #     ranks.append(core.shape[0])
    # ranks.append(ranks[0])

    X_tube = X + E

    tr_cores = tr.tr_decompose2(X_tube, G_list, max_iter=iner_num_steps, tol=inner_tol, dis_num = 2000, device="cuda")
    # tr_cores = tr.tr_decompose(X_tube, ranks, max_iter=iner_num_steps, tol=inner_tol, dis_num = 2000, device="cuda")

    # del X_tube
    # torch.cuda.empty_cache()

    # # 确保 tr_cores 启用了梯度追踪
    # for core in tr_cores:
    #     core.requires_grad_(True)

    TN_G = tl.tr_to_tensor(tr_cores)  # Reconstruct tensor from tensor train
    # # 在计算损失之前检查 TN_G 的状态
    # print("TN_G requires_grad:", TN_G.requires_grad)
    # print("X requires_grad:", X.requires_grad)
    # print("E requires_grad:", E.requires_grad)
    # optimizer = torch.optim.Adam([E], lr=config.lr)
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
    # optimizer = torch.optim.Adam([E], lr=config.lr)

    # Example: Use a different decomposition method or a different loss function
    EC = E.clone().detach()
    tr_cores = tr.tr_decompose2(X+EC, G_list, max_iter=inner_num_steps, tol=inner_tol, dis_num = 2000, device="cuda")
    TN_G = tl.tr_to_tensor(tr_cores)  # Reconstruct tensor from tensor train

    loss_values = []

    return -torch.norm(X+E - TN_G) ** 2, TN_G, tr_cores  # Example: Minimize the norm instead of maximizing

def objective_function_ATR_core(X, G_list, inner_num_steps, inner_tol):
    """
    Compute the comparison objective function.
    X: Original tensor (PyTorch tensor)
    G_list: List of factor tensors in tensor train format
    E: Perturbation tensor (PyTorch tensor)
    inner_num_steps: Number of inner optimization steps
    inner_tol: Tolerance for inner optimization
    """
    # optimizer = torch.optim.Adam([E], lr=config.lr)

    # Example: Use a different decomposition method or a different loss function
    
    tr_cores = tr.tr_decompose2(X, G_list, max_iter=inner_num_steps, tol=inner_tol, dis_num = 2000, device="cuda")
    TN_G = tl.tr_to_tensor(tr_cores)  # Reconstruct tensor from tensor train

    loss_values = []

    return -torch.norm(X - TN_G) ** 2, TN_G, tr_cores  # Example: Minimize the norm instead of maximizing

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
    # E_unfold = tr.generalized_k_unfolding(E, new_X_dims, 0)
    X_unfold = torch.tensor(X_unfold, requires_grad=True, dtype=torch.float32).to('cuda')
    # E_unfold = torch.tensor(E_unfold, requires_grad=True, dtype=torch.float32).to('cuda')

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

        # rse_E = torch.norm(E - E_old)

        norm2_X_TN_G = torch.norm(X - TN_G).item()
        rec_loss_values.append(norm2_X_TN_G)
        norm2_X_TN_G_list.append(norm2_X_TN_G)

        norm_E = torch.norm(E)
        # print(f"Comparison Step {step}, Loss: {norm2_X_TN_G}, normE: {norm_E}")
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
        # E.data = config.epsilon * E / torch.norm(E)
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

            # E_unfold = X_unfold @ torch.linalg.pinv(Gr_unfold) @ Gr_unfold - X_unfold
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

        # # 检查梯度
        # print("梯度在反向传播后的值：", E.grad)

        # Update E
        optimizer.step()

        # Apply the constraint ||E||_F^2 <= config.epsilon
        scaling_factor = (config.epsilon ** 0.5) / torch.norm(E)
        if torch.norm(E)**2 > config.epsilon:
            E.data = scaling_factor * E

        rse_E = torch.norm(E - E_old)
        # print(f"Step {step}, RSE_E: {rse_E}")

        # loss_sqrt = math.sqrt(abs(loss))
        # loss_values.append(loss_sqrt)
        norm2_X_TN_G = torch.norm(X - TN_G).item()
        rec_loss_values.append(norm2_X_TN_G)

        norm_E = torch.norm(E)
        # print(f"Comparison Step {step}, Loss: {norm2_X_TN_G}, normE: {norm_E}")
        logging.info(f"Comparison Step {step}, Loss_X_TNG: {norm2_X_TN_G}, normE: {norm_E}")

        if step != 0 and step % config.outer_dis_num == 0:
            print(f"Comparison Step {step}, Loss: {norm2_X_TN_G}, normE: {norm_E}")

        # X = X.detach()
        # E = E.detach()
        # TN_G = TN_G.detach()
        # G_list = [G.detach() for G in G_list]
            

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
    # X.requires_grad_(True)
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
        # loss2, TN_G = objective_function_ATR(X, G_list, E, config.inner_num_steps, config.inner_tol)
        EC = E.clone().detach()
        tr_cores = tr.tr_decompose2(X+EC, G_list, max_iter=inner_num_steps, tol=inner_tol, dis_num = 2000, device="cuda")
        TN_G = tl.tr_to_tensor(tr_cores)  # Reconstruct tensor from tensor train
        # loss2 = -torch.norm(X +E - TN_G) ** 2
        E = E - config.lr * 2 * (X + E - TN_G)

        E_old = E.clone().detach()


        # Apply the constraint ||E||_F^2 <= config.epsilon
        scaling_factor = (config.epsilon ** 0.5) / torch.norm(E)
        if torch.norm(E)**2 > config.epsilon:
            E.data = scaling_factor * E

        rse_E = torch.norm(E - E_old)
        # print(f"Step {step}, RSE_E: {rse_E}")

        # loss_sqrt = math.sqrt(abs(loss))
        # loss_values.append(loss_sqrt)
        norm2_X_TN_G = torch.norm(X - TN_G).item()
        rec_loss_values.append(norm2_X_TN_G)

        norm_E = torch.norm(E)
        # print(f"Comparison Step {step}, Loss: {norm2_X_TN_G}, normE: {norm_E}")
        logging.info(f"Comparison Step {step}, Loss_X_TNG: {norm2_X_TN_G}, normE: {norm_E}")

        if step != 0 and step % config.outer_dis_num == 0:
            print(f"Comparison Step {step}, Loss: {norm2_X_TN_G}, normE: {norm_E}")

        # X = X.detach()
        # E = E.detach()
        # TN_G = TN_G.detach()
        # G_list = [G.detach() for G in G_list]
            

    return E, TN_G, rec_loss_values

def optimize_perturbation_comparison_noise_rand(X, G_list, config: OptimizationConfig):
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


        E = torch.randn_like(X, requires_grad=True, device='cuda')

        # Apply the constraint ||E||_F^2 <= config.epsilon
        scaling_factor = (config.epsilon ** 0.5) / torch.norm(E)
        if torch.norm(E)**2 > config.epsilon:
            E.data = scaling_factor * E

        rse_E = torch.norm(E - E_old)
        
        norm2_X_TN_G = torch.norm(X - TN_G).item()
        rec_loss_values.append(norm2_X_TN_G)

        norm_E = torch.norm(E)

        if step != 0 and step % config.outer_dis_num == 0:
            print(f"Comparison_2 Step {step}, Loss: {norm2_X_TN_G}, normE: {norm_E}")
            

    return E, TN_G, rec_loss_values

def optimize_perturbation_comparison_noise_rand_core(X, G_list, config: OptimizationConfig):
    """
    Optimize perturbation tensor \mathcal{E} using gradient descent for comparison.
    X: Original tensor (PyTorch tensor)
    G_list: List of factor tensors in tensor train format
    config: Optimization configuration
    """
    # Initialize perturbation tensor \mathcal{E}
    # E = torch.randn_like(X, requires_grad=True, device='cuda')
    E_list = [torch.randn_like(G) for G in G_list]
    norm_squared = sum(torch.norm(E)**2 for E in E_list)
    epsilon = config.epsilon /10
    if norm_squared > epsilon:
        scaling_factor = (epsilon**0.5 / torch.sqrt(norm_squared))
        E_list = [E * scaling_factor for E in E_list]
    X = X.detach()
    # X.requires_grad_(True)
    G_list = [G.detach() for G in G_list]
    # Apply the constraint ||E||_F^2 <= epsilon
    E = torch.randn_like(X, requires_grad=True, device='cuda')

    optimizer = torch.optim.Adam([E], lr=config.lr)

    rec_loss_values = []

    G_plus_E_list = [G.detach() for G in G_list]
    # 将每个 G 和 E 相加
    # G_plus_E_list = [G + E for G, E in zip(G_list, E_list)]
    for step in range(config.outer_num_steps):
        optimizer.zero_grad()

        # Compute loss
        loss2, TN_G, G_plus_E_list = objective_function_ATR_core(X, G_plus_E_list, config.inner_num_steps, config.inner_tol)

        norm2_X_TN_G = torch.norm(X - TN_G).item()

        if norm2_X_TN_G < config.inner_tol * 100:
            E_list = [torch.randn_like(G) for G in G_list]
            norm_squared = sum(torch.norm(E)**2 for E in E_list)
            if norm_squared > epsilon:
                scaling_factor = (epsilon**0.5 / torch.sqrt(norm_squared))
                E_list = [E * scaling_factor for E in E_list]

            G_plus_E_list = [G + E for G, E in zip(G_list, E_list)]
        
        rec_loss_values.append(norm2_X_TN_G)

        norm_E = norm_squared

        if step != 0 and step % config.outer_dis_num == 0:
            print(f"Comparison_3 Step {step}, Loss: {norm2_X_TN_G}, normE: {norm_E}")
            

    return E_list, TN_G, rec_loss_values

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
    # output_folder2 = 'Results My picture2'
    # if not os.path.exists(output_folder2):
    #     os.makedirs(output_folder2)
    # 创建一个字典来存储所有的核心张量
    tr_cores_dict = {f'tr_core_{i}': tr_cores_np[i] for i in range(len(tr_cores_np))}


    X = X.detach()
    tr_cores = [G.detach() for G in tr_cores]
    # tr_cores2 = tr.tr_decompose2(X, tr_cores, max_iter=inner_num_steps*outer_num_steps, tol=inner_tol, device="cuda")
    # TN_G_ori = tl.tr_to_tensor(tr_cores2) 
    # norm2_original_TR_algorithm = torch.norm(X - TN_G_ori).item()

    # E_Gaus = torch.normal(mean=0, std=1, size=X.shape, device="cuda")
    # # Apply the constraint ||E||_F^2 <= epsilon
    # scaling_factor = (config.epsilon ** 0.5) / torch.norm(E_Gaus)
    # if torch.norm(E_Gaus)**2 > config.epsilon:
    #     E_Gaus.data = scaling_factor * E_Gaus
    # tr_cores3 = tr.tr_decompose2(X+E_Gaus, tr_cores, max_iter=inner_num_steps*outer_num_steps, tol=inner_tol, device="cuda")
    # TN_G_Gaus = tl.tr_to_tensor(tr_cores3) 
    # norm2_Gaus = torch.norm(X - TN_G_Gaus).item()


    # print(f"Norm2 (Original TR Algorithm): {norm2_original_TR_algorithm}")
    # logging.info(f"Norm2 (Original TR Algorithm): {norm2_original_TR_algorithm}")
    # print(f"Norm2 (Gaussian Noise): {norm2_Gaus}")
    # logging.info(f"Norm2 (Gaussian Noise): {norm2_Gaus}")

    X_nuclear_norm = torch.linalg.norm(X.cpu().detach(), ord='nuc')
    if compare_method_enabled:
        # Optimize perturbation \mathcal{E} using comparison algorithm
        start_time = time.time()
        # config.outer_num_steps = 100
        E_comparison, TN_G_comparison, loss_ATR = optimize_perturbation_comparison(X, tr_cores, config)
        time_comp = time.time() - start_time
        print(f"time_comp 运行时间: {time_comp:.4f} 秒")
        norm2_comparison_algorithm = torch.norm(X - TN_G_comparison).item()
        print(f"Norm2 (Comparison Algorithm): {norm2_comparison_algorithm}")
        logging.info(f"Norm2 (Comparison Algorithm): {norm2_comparison_algorithm}")
        X_cpu = X.cpu().detach().numpy()
        E_comparison_cpu = E_comparison.cpu().detach().numpy()
        X_E_comparison_cpu = (X + E_comparison).cpu().detach().numpy()
        TN_G_comparison = (TN_G_comparison).cpu().detach().numpy()

        # 计算 X 的核范数
        # X_nuclear_norm = torch.linalg.norm(X_cpu, ord='nuc')
        X_plus_E_comparison_nuclear_norm = torch.linalg.norm(X_E_comparison_cpu, ord='nuc')


        # 将 E_comparison_cpu 展平为一维数组
        E_comparison_flat = E_comparison_cpu.flatten()

        # 绘制直方图和密度图
        plt.figure(figsize=(10, 6))

        # 直方图
        sns.histplot(E_comparison_flat, bins=50, kde=True, color='blue', stat='density')

        # 设置图形标题和标签
        plt.title('Distribution of E_comparison')
        plt.xlabel('Value')
        plt.ylabel('Density')

        # 显示图形
        plt.show()

        plt.savefig(f'{output_folder}/Distribution_E_{base_name}.png')

        start_time = time.time()
        # config.outer_num_steps = 100
        E_comparison2, TN_G_comparison2, loss_ATR = optimize_perturbation_comparison_noise_rand(X, tr_cores, config)
        time_comp = time.time() - start_time
        print(f"time_comp 运行时间: {time_comp:.4f} 秒")
        norm2_comparison_algorithm2 = torch.norm(X - TN_G_comparison2).item()
        print(f"Norm2 (Comparison Algorithm): {norm2_comparison_algorithm2}")
        logging.info(f"Norm2 (Comparison Algorithm): {norm2_comparison_algorithm2}")
        E_comparison_cpu2 = E_comparison2.cpu().detach().numpy()
        X_E_comparison_cpu2 = (X + E_comparison2).cpu().detach().numpy()
        TN_G_comparison2 = (TN_G_comparison2).cpu().detach().numpy()

        start_time = time.time()
        # config.outer_num_steps = 100
        E_comparison3, TN_G_comparison3, loss_ATR = optimize_perturbation_comparison_noise_rand_core(X, tr_cores, config)
        time_comp = time.time() - start_time
        print(f"time_comp 运行时间: {time_comp:.4f} 秒")
        norm2_comparison_algorithm3 = torch.norm(X - TN_G_comparison3).item()
        print(f"Norm2 (Comparison Algorithm): {norm2_comparison_algorithm3}")
        logging.info(f"Norm2 (Comparison Algorithm): {norm2_comparison_algorithm3}")
        E_comparison_cpu3 = E_comparison3.cpu().detach().numpy()
        X_E_comparison_cpu3 = (X + E_comparison3).cpu().detach().numpy()
        TN_G_comparison3 = (TN_G_comparison3).cpu().detach().numpy()

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

        # 显示 TN_G 图像
        plt.clf()
        plt.imshow(X_E_cpu)
        savemat(os.path.join(output_folder, f'X_E_Ours_{base_name}.mat'), {f'X_E_Ours_{base_name}': X_E_cpu})
        # plt.title('TN_G_Ours')

        # 保存为 JPEG 格式
        plt.savefig(os.path.join(output_folder, f'X_E_Ours_{base_name}.jpeg'), format='jpeg', bbox_inches='tight', pad_inches=0)
        plt.axis('off')

        # 保存图像os.path.join(output_folder, f'{base_name}_X_E.png')
        plt.savefig(os.path.join(output_folder, f'X_E_Ours_{base_name}.png'), bbox_inches='tight', pad_inches=0)

        plt.imshow(X_E_cpu_F)
        savemat(os.path.join(output_folder, f'X_E_Ours_F_{base_name}.mat'), {f'X_E_Ours_F_{base_name}': X_E_cpu_F})
        plt.axis('off')
        plt.savefig(os.path.join(output_folder, f'X_E_Ours_F_{base_name}.png'), bbox_inches='tight', pad_inches=0)
        plt.savefig(os.path.join(output_folder, f'X_E_Ours_F_{base_name}.jpeg'), format='jpeg', bbox_inches='tight', pad_inches=0)
       
        # 显示 TN_G 图像
        plt.imshow(X_cpu)
        savemat(os.path.join(output_folder, f'X_{base_name}.mat'), {f'X_{base_name}': X_cpu})
        # plt.title('TN_G_ori_output')
        plt.axis('off')

        # 保存图像
        plt.savefig(os.path.join(output_folder, f'X_{base_name}.png'), bbox_inches='tight', pad_inches=0)
        plt.savefig(os.path.join(output_folder, f'X_{base_name}.jpeg'), format='jpeg', bbox_inches='tight', pad_inches=0)
       

        plt.clf()
        plt.imshow(X_E_cpu_Gaus)
        savemat(os.path.join(output_folder, f'X_E_Gaus_{base_name}.mat'), {f'X_E_Gaus_{base_name}': X_E_cpu_Gaus})
        # plt.title('TN_G_Ours')
        plt.axis('off')

        # 保存图像os.path.join(output_folder, f'{base_name}_X_E.png')
        plt.savefig(os.path.join(output_folder, f'X_E_Gaus_{base_name}.png'), bbox_inches='tight', pad_inches=0)
        plt.savefig(os.path.join(output_folder, f'X_E_Gaus_{base_name}.jpeg'), format='jpeg', bbox_inches='tight', pad_inches=0)
       

    # 获取当前时间并格式化为字符串
    # output_folder3 = 'Results My picture comparison'
    # if not os.path.exists(output_folder3):
    #     os.makedirs(output_folder3)
    # current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_folder, f'comparison_{base_name}.png')
    if compare_method_enabled:
        
        # 显示 TN_G 图像
        plt.imshow(X_E_comparison_cpu)
        savemat(os.path.join(output_folder,f'X_E_comparison_{base_name}.mat'), {f'X_E_comparison_{base_name}': X_E_comparison_cpu})
        # plt.title('TN_G_comp_output')
        plt.axis('off')

        # 保存图像
        plt.savefig(os.path.join(output_folder, f'X_E_comparison_{base_name}.png'), bbox_inches='tight', pad_inches=0)
        plt.savefig(os.path.join(output_folder, f'X_E_comparison_{base_name}.jpeg'), format='jpeg', bbox_inches='tight', pad_inches=0)

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
        plt.savefig(f'{output_folder}/function_time_comparison_{base_name}.png')
        plt.show()

    return time_simplified

# Example usage
if __name__ == "__main__":
    # Example tensor X (3D for simplicity)
    # 读取 Lena 图片
    # lena = Image.open('lena.png')
    input_folder = './AT_TD/My picture'
    log_folder = 'Resluts My picture7'
    output_folder = os.path.join(log_folder, 'output')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # lena = Image.open('./AT_TD/0081x2.png')  # 去除 Alpha 通道
    # lena = np.array(lena)
    # X = torch.tensor(lena/255.0, dtype=torch.float32).to("cuda")
 
    # Decompose X into tensor train format
    ranks = [5, 5, 5]  # Rank for each factor tensor
    ranks = ranks + [ranks[0]]  # 最后一个 rank 应该等于第一个 rank
    # tr_cores = tr.tr_decompose(X, ranks, max_iter=1, tol=1e-10, device="cuda")
    # shape = list(X.shape)
    # tr_cores = tr.initialize_tr_cores(shape, ranks, device="cuda")
    # tr_cores = [tr_core.requires_grad_(True) for tr_core in tr_cores]

    # Set input parameter 
    epsilon = (0.5)**2 # Frobenius norm constraint for perturbation tensor
    logging.info(f"Norm2 (epsilon): {epsilon}")
    inner_tol = 1e-10 # Tolerance for inner optimization
    lr = 0.01  # Learning rate for gradient descent
    outer_num_steps = 100 # Number of outer optimization steps
    inner_num_steps = 200 # Number of inner optimization steps
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
    for i in range(1, 2):
        image_name = f'{i}.png'
        image_path = os.path.join(input_folder, image_name)
        if os.path.isfile(image_path):
            time_simplified = process_image(image_path, output_folder, config)






