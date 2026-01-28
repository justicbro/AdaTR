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

# 指定使用第 0 张 GPU 卡
os.environ["CUDA_VISIBLE_DEVICES"] = "9"

# 定义一个布尔变量来控制日志记录
log_enabled = False
imshow_enabled = True
compare_method_enabled = False

# 创建日志文件夹
if log_enabled:
    log_folder = 'logs'
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

def display_images(original_image, images, titles, rows, cols, figsize=(20, 10), save_path=None):
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        psnr = calculate_psnr(original_image, images[i])
        ax.set_title(f"{titles[i]}\nPSNR: {psnr:.2f} dB")
        ax.axis('off')
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
    # ranks = []
    # for core in G_list:
    #     ranks.append(core.shape[0])
    # ranks.append(ranks[0])

    X_tube = X + E

    tr_cores = tr.tr_decompose2(X_tube, G_list, max_iter=inner_num_steps, tol=inner_tol, dis_num = 2000, device="cuda")
    # tr_cores = tr.tr_decompose(X_tube, ranks, max_iter=inner_num_steps, tol=inner_tol, dis_num = 2000, device="cuda")

    del X_tube
    torch.cuda.empty_cache()

    # 确保 tr_cores 启用了梯度追踪
    for core in tr_cores:
        core.requires_grad_(True)

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
    optimizer = torch.optim.Adam([E], lr=config.lr)

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



def optimize_perturbation_simplifed(X, G_list, config: OptimizationConfig):
    rse_E_list = []
    loss_list = []

    # Initialize perturbation tensor \mathcal{E}
    E = torch.randn_like(X, requires_grad=True, device='cuda')

    # 计算每个维度的权重
    total_size = sum(X.size())
    weights = [X.size(dim) / total_size for dim in range(X.dim())]

    # 初始化 E 的梯度为零
    E_grad = torch.zeros_like(E)
    
    # Apply the constraint ||E||_F^2 <= epsilon
    scaling_factor = (config.epsilon ** 0.5) / torch.norm(E)
    if torch.norm(E)**2 > config.epsilon:
        # E.data = config.epsilon * E / torch.norm(E)
        E.data = (scaling_factor) * E

    for step in range(config.outer_num_steps):
        G_list = tr.tr_decompose2(X+E, G_list, max_iter=inner_num_steps, tol=inner_tol, dis_num=config.inner_dis_num, device="cuda")
        TN_G = tl.tr_to_tensor(G_list)  # Reconstruct tensor from tensor train

        E_old = E.clone().detach()

        # 初始化 E 的梯度为零
        E_grad = torch.zeros_like(E)
        E_sum = torch.zeros_like(E)
        # 遍历每个维度，计算梯度并累加
        for n in range(X.dim()):
            X_dims= list(range(X.dim()))
            new_X_dims = [X_dims[n]] + X_dims[n+1:] + X_dims[:n]
            X_unfold = tr.generalized_k_unfolding(X, new_X_dims, 0)
            X_unfold = torch.tensor(X_unfold, requires_grad=True, dtype=torch.float32).to('cuda')

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
        # n = X.dim() - 2
        # X_dims= list(range(X.dim()))
        # new_X_dims = [X_dims[n]] + X_dims[n+1:] + X_dims[:n]
        # X_unfold = tr.generalized_k_unfolding(X, new_X_dims, 0)
        # X_unfold = torch.tensor(X_unfold, requires_grad=True, dtype=torch.float32).to('cuda')

        # # 计算除 cores[n] 之外的张量积
        # X_hat = tr.contract_except_n(G_list, n)
        # dims = list(range(X_hat.dim()))
        # new_dims = [dims[-1]] + dims[:-1]
        # Gr_unfold = tr.generalized_k_unfolding(X_hat, new_dims, 1)

        # # E_unfold = X_unfold @ torch.linalg.pinv(Gr_unfold) @ Gr_unfold - X_unfold
        # config.lr = 0.1
        # E_unfold = tr.generalized_k_unfolding(E, new_X_dims, 0)
        # E_unfold_grad = (X_unfold - (X_unfold + E_unfold) @ torch.linalg.pinv(Gr_unfold) @ Gr_unfold) @ Gr_unfold.T @ torch.linalg.pinv(Gr_unfold).T
        # E_unfold = E_unfold - config.lr * E_unfold_grad

        # E_folded = E_unfold.reshape(*[X.shape[dim] for dim in new_X_dims])
        # E = E_folded.permute(*[new_X_dims.index(i) for i in range(len(new_X_dims))])

        # Apply the constraint ||E||_F^2 <= epsilon
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
        # print(f"Step {step}, RSE_E: {rse_E}, Loss: {loss_X_TNG}, normE: {norm_E}")

        rse_E_list.append(rse_E)
        if step % config.outer_dis_num == 0:
            print(f"Step {step}, RSE_E: {rse_E}, Loss: {loss_X_TNG}, normE: {norm_E}")
        # if step % config.outer_dis_num == 0:
        #     print(f"Step {step}, Loss: {loss.item()}")

    return E, TN_G, loss_list

def optimize_perturbation(X, G_list, config: OptimizationConfig):
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
            
        # Compute loss
        loss, TN_G, G_list = objective_function_ATTR(X, G_list, E, config.inner_num_steps, config.inner_tol)

        # print(f"TN_G max: {TN_G.max()}, min: {TN_G.min()}")

        # 检查损失是否为 NaN
        if torch.isnan(loss):
            raise ValueError("Loss is NaN")
    
        # grads = torch.autograd.grad(loss, E, retain_graph=True, allow_unused=True)
        # print(f"Gradient of E: {grads}")

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_([E], max_norm=0.1)

        # 在反向传播之前检查 E 的梯度
        if E.grad is not None:
            print("反向传播之前的梯度：", E.grad)
        # Backpropagation
        loss.backward(retain_graph=True)
        # print(loss.requires_grad)  # 确保损失函数需要梯度

        # 检查梯度
        # print("梯度在反向传播后的值：", E.grad)
        print(f"Gradient max: {E.grad.max()}, min: {E.grad.min()}")
        if torch.isnan(E.grad).any():
            break

        # Update E
        optimizer.step()

        # print(f"E max: {E.max()}, min: {E.min()}")

        E_plus_X = X + E
        E_plus_X_clamp = torch.clamp(E_plus_X, 0, 1)
        E.data = E_plus_X_clamp - X

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

        TN_G2 = TN_G.clone().detach()

        # 释放计算图
        del loss, TN_G
        torch.cuda.empty_cache()
            

    return E, TN_G2, loss_list

def optimize_perturbation_comparison(X, G_list, config: OptimizationConfig):
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

    optimizer = torch.optim.Adam([E], lr=config.lr)

    rec_loss_values = []

    for step in range(config.outer_num_steps):
        optimizer.zero_grad()

        # Compute loss
        loss, TN_G, G_list = objective_function_ATR(X, G_list, E, config.inner_num_steps, config.inner_tol)

        E_old = E.clone().detach()

        # Backpropagation
        loss.backward(retain_graph=True)

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
            

    return E, TN_G, rec_loss_values

# Example usage
if __name__ == "__main__":
    # Example tensor X (3D for simplicity)
    # 读取 Lena 图片
    lena = Image.open('lena.png')
    lena = np.array(lena)
    X = torch.tensor(lena/255.0, dtype=torch.float32).to("cuda")
 
    # Decompose X into tensor train format
    ranks = [5, 5, 5]  # Rank for each factor tensor
    ranks = ranks + [ranks[0]]  # 最后一个 rank 应该等于第一个 rank
    # tr_cores = tr.tr_decompose(X, ranks, max_iter=1, tol=1e-10, device="cuda")
    shape = list(X.shape)
    tr_cores = tr.initialize_tr_cores(shape, ranks, device="cuda")

    # Set input parameter 
    # epsilon = (0.1)**2 # Frobenius norm constraint for perturbation tensor
    # 定义不同的 epsilon 值
    epsilon_values = [0.1, 0.5, 1.0, 5, 10, 50, 100, 500, 1000, 5000, 10000]
    # 初始化损失列表
    loss_ours = []
    loss_comparison = []
    
    inner_tol = 1e-10 # Tolerance for inner optimization
    lr = 0.01  # Learning rate for gradient descent
    outer_num_steps = 50 # Number of outer optimization steps
    inner_num_steps = 500 # Number of inner optimization steps
    outer_dis_num = 1 # Display loss every outer_dis_num steps
    inner_dis_num = 2000 # Display loss every inner_dis_num steps
    logging.info(f"Optimization parameters:")
    # logging.info(f"epsilon (Frobenius norm constraint): {epsilon}")
    logging.info(f"inner_tol (Tolerance for inner optimization): {inner_tol}")
    logging.info(f"lr (Learning rate): {lr}")
    logging.info(f"outer_num_steps (Number of outer optimization steps): {outer_num_steps}")
    logging.info(f"inner_num_steps (Number of inner optimization steps): {inner_num_steps}")
    logging.info(f"outer_dis_num (Display loss every outer_dis_num steps): {outer_dis_num}")
    logging.info(f"inner_dis_num (Display loss every inner_dis_num steps): {inner_dis_num}")

    

    # Optimize perturbation \mathcal{E}
    X.requires_grad_(True)
    for epsilon in epsilon_values:
        config = OptimizationConfig(epsilon, lr, outer_num_steps, inner_num_steps, outer_dis_num, inner_dis_num, inner_tol)
        logging.info(f"Norm2 (epsilon): {epsilon}")
        config.epsilon = epsilon  # Frobenius norm constraint for perturbation tensor

        # 运行 optimize_perturbation
        E_optimized, TN_G, loss_ours_list = optimize_perturbation(X, tr_cores, config)
        loss_ours.append(loss_ours_list[-1])  # 记录最后一步的损失

        # 运行 optimize_perturbation_comparison
        E_optimized_comp, TN_G_comp, loss_comparison_list = optimize_perturbation_comparison(X, tr_cores, config)
        loss_comparison.append(loss_comparison_list[-1])  # 记录最后一步的损失
        

    tr_cores = tr.tr_decompose(X, ranks, max_iter=inner_num_steps, tol=1e-10, device="cuda")
    TN_G_ori = tl.tr_to_tensor(tr_cores) 
    norm2_original_TR_algorithm = torch.norm(X - TN_G_ori).item()

# 绘制折线图
plt.figure(figsize=(10, 6))
plt.plot(epsilon_values, loss_ours, label='Ours', marker='o')
plt.plot(epsilon_values, loss_comparison, label='Comparison', marker='x')
plt.xscale('log')  # 使用对数刻度
plt.yscale('log')  # 使用对数刻度
plt.xlabel('Epsilon')
plt.ylabel('Loss')
plt.title('Loss vs Epsilon')
plt.legend()
plt.grid(True)

# 保存图像
plt.savefig('loss_vs_epsilon.png')

# 显示图像
plt.show()
    