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
import NMF_func as mynmf


# 定义一个布尔变量来控制日志记录
log_enabled = False
imshow_enabled = True
compare_method_enabled = True

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
        ax.imshow(images[i], cmap='gray')
        psnr = calculate_psnr(original_image, images[i])
        ax.set_title(f"{titles[i]}\nPSNR: {psnr:.2f} dB")
        ax.axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.show()

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

def objective_function_ATNMF(X, rank, E, inner_num_steps = 100, inner_tol = 1e-10, inner_dis_num = 100, device='cuda'):
    """
    Compute the objective function ||X - TN([G^(k)])||_F^2.
    X: Original tensor (PyTorch tensor)
    G_list: List of factor tensors in tensor train format
    E: Perturbation tensor (PyTorch tensor)
    """
    # lambda2 = 5
    lambda2 = 0

    X_tube = X + E

    W, H = mynmf.nmf(X_tube, rank, inner_num_steps, inner_tol, inner_dis_num, device=device)

    V_reconstructed = torch.mm(W, H)

    # 计算多个 X_tube 的核范数
    nuclear_norm_X_tube = torch.norm(X_tube, p='nuc')
    
    return -torch.norm(X - V_reconstructed) ** 2 - lambda2 * nuclear_norm_X_tube, V_reconstructed

def objective_function_ANMF(X, rank, E, inner_num_steps = 100, inner_tol = 1e-10, inner_dis_num = 100, device='cuda'):
    """
    Compute the comparison objective function.
    X: Original tensor (PyTorch tensor)
    G_list: List of factor tensors in tensor train format
    E: Perturbation tensor (PyTorch tensor)
    inner_num_steps: Number of inner optimization steps
    inner_tol: Tolerance for inner optimization
    """
    EC = E.clone().detach()
    X_tube = X + EC

    W, H = mynmf.nmf(X_tube, rank, inner_num_steps, inner_tol, inner_dis_num, device=device)

    V_reconstructed = torch.mm(W, H)
    
    return -torch.norm(X + E - V_reconstructed) ** 2, V_reconstructed


def optimize_perturbation(X, rank, config: OptimizationConfig):
    """
    Optimize perturbation tensor \mathcal{E} using gradient descent.
    X: Original tensor (PyTorch tensor)
    G_list: List of factor tensors in tensor train format
    epsilon: Frobenius norm constraint for \mathcal{E}
    lr: Learning rate for gradient descent
    num_steps: Number of gradient descent steps
    """
    # Initialize perturbation tensor \mathcal{E}
    E = torch.randn_like(X, requires_grad=True, device='cuda')
    # Apply the constraint ||E||_F^2 <= epsilon
    scaling_factor = config.epsilon ** 0.5 / torch.norm(E)
    if torch.norm(E) ** 2 > config.epsilon:
        E.data = scaling_factor * E 

    loss_values = []

    optimizer = torch.optim.Adam([E], lr=config.lr)

    for step in range(config.outer_num_steps):
        optimizer.zero_grad() # Clear gradients

        E_old = E.clone().detach()

        X.requires_grad_(True)
        
        # Compute loss
        loss, TN_G = objective_function_ATNMF(X, rank, E, config.inner_num_steps, config.inner_tol, config.inner_dis_num, device='cuda')

        grads = torch.autograd.grad(loss, E, retain_graph=True, allow_unused=True)
        # print(f"Gradient of E: {grads}")
        # Backpropagation
        loss.backward()
        # print(loss.requires_grad)  # 确保损失函数需要梯度

        # # 检查梯度
        # print("梯度在反向传播后的值：", E.grad)

        # Update E
        optimizer.step()

        # Apply the constraint ||E||_F^2 <= epsilon
        scaling_factor = config.epsilon ** 0.5 / torch.norm(E)
        if torch.norm(E) ** 2 > config.epsilon:
            E.data = scaling_factor * E 

        # rse_E = torch.norm(E - E_old)
        # print(f"Step {step}, RSE_E: {rse_E}")

        # 记录 norm2 结果到日志
        loss_sqrt = math.sqrt(abs(loss))
        loss_values.append(loss_sqrt)

        norm2_E = torch.norm(E.data).item()
        logging.info(f"Step {step} norm(X - TN_G_Ours): {loss_sqrt}, norm2(E): {norm2_E}")

        if step % config.outer_dis_num == 0:
            print(f"Step {step}, Loss: {loss.item()}, norm2_E: {norm2_E}")
            
            

    return E, TN_G, loss_values

def optimize_perturbation_comparison(X, rank, config: OptimizationConfig):
    """
    Optimize perturbation tensor \mathcal{E} using gradient descent for comparison.
    X: Original tensor (PyTorch tensor)
    G_list: List of factor tensors in tensor train format
    config: Optimization configuration
    """
    # Initialize perturbation tensor \mathcal{E}
    E = torch.randn_like(X, requires_grad=True, device='cuda')
    # Apply the constraint ||E||_F^2 <= epsilon
    scaling_factor = config.epsilon ** 0.5 / torch.norm(E)
    if torch.norm(E) ** 2 > config.epsilon:
        E.data = scaling_factor * E 

    optimizer = torch.optim.Adam([E], lr=config.lr)

    rec_loss_values = []

    for step in range(config.outer_num_steps):
        optimizer.zero_grad()

        # Compute loss
        loss, TN_G = objective_function_ANMF(X, rank, E, config.inner_num_steps, config.inner_tol, config.inner_dis_num, device='cuda')

        E_old = E.clone().detach()

        # Backpropagation
        loss.backward()

        # # 检查梯度
        # print("梯度在反向传播后的值：", E.grad)

        # Update E
        optimizer.step()

        # Apply the constraint ||E||_F^2 <= config.epsilon
        scaling_factor = config.epsilon ** 0.5 / torch.norm(E)
        if torch.norm(E) ** 2 > config.epsilon:
            E.data = scaling_factor * E 

        # rse_E = torch.norm(E - E_old)
        # print(f"Step {step}, RSE_E: {rse_E}")

        # loss_sqrt = math.sqrt(abs(loss))
        # loss_values.append(loss_sqrt)
        norm2_X_TN_G = torch.norm(X - TN_G).item()
        rec_loss_values.append(norm2_X_TN_G)

        norm2_E = torch.norm(E.data).item()
        logging.info(f"Step {step} norm(X - TN_G_ATR): {norm2_X_TN_G}, norm2(E): {norm2_E}")

        if step != 0 and step % config.outer_dis_num == 0:
            print(f"Comparison Step {step}, Loss: {loss.item()}, norm2_E: {norm2_E}")
            

    return E, TN_G, rec_loss_values

# Example usage
if __name__ == "__main__":

    lena = Image.open('lena.png').convert('L')
    V = np.array(lena, dtype=np.float32)  # Ensure the dtype is float32

    # 将 V 转换为 PyTorch 张量并移动到 GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = torch.tensor(V, device=device)/255
    

    # Decompose X into tensor train format
    ranks = 50
    # tr_cores = tr.tr_decompose(X, ranks, max_iter=1, tol=1e-10, device="cuda")


    # Set input parameter 
    epsilon = 0.1 ** 2# Frobenius norm constraint for perturbation tensor
    inner_tol = 1e-10 # Tolerance for inner optimization
    lr = 0.0001  # Learning rate for gradient descent
    outer_num_steps = 300 # Number of outer optimization steps
    inner_num_steps = 100 # Number of inner optimization steps
    outer_dis_num = 10 # Display loss every outer_dis_num steps
    inner_dis_num = 2000 # Display loss every inner_dis_num steps
    logging.info(f"Proposed-ATNMF Experiment")
    logging.info(f"Optimization parameters:")
    logging.info(f"epsilon (Frobenius norm constraint): {epsilon}")
    logging.info(f"inner_tol (Tolerance for inner optimization): {inner_tol}")
    logging.info(f"lr (Learning rate): {lr}")
    logging.info(f"outer_num_steps (Number of outer optimization steps): {outer_num_steps}")
    logging.info(f"inner_num_steps (Number of inner optimization steps): {inner_num_steps}")
    logging.info(f"outer_dis_num (Display loss every outer_dis_num steps): {outer_dis_num}")
    logging.info(f"inner_dis_num (Display loss every inner_dis_num steps): {inner_dis_num}")

    config = OptimizationConfig(epsilon, lr, outer_num_steps, inner_num_steps, outer_dis_num, inner_dis_num, inner_tol)

    # Optimize perturbation \mathcal{E}
    X.requires_grad_(True)
    E_optimized, TN_G, loss_ours = optimize_perturbation(X, ranks, config)

    # 计算 norm2
    norm2_our_algorithm = torch.norm(X - TN_G).item()
    # 打印 norm2 结果
    print(f"Norm2 (Our Algorithm): {norm2_our_algorithm}")
    

    if compare_method_enabled:
        # Optimize perturbation \mathcal{E} using comparison algorithm
        E_comparison, TN_G_comparison, loss_ATR = optimize_perturbation_comparison(X, ranks, config)
        norm2_comparison_algorithm = torch.norm(X - TN_G_comparison).item()
        print(f"Norm2 (Comparison Algorithm): {norm2_comparison_algorithm}")
        logging.info(f"Norm2 (Comparison Algorithm): {norm2_comparison_algorithm}")
        E_comparison_cpu = E_comparison.cpu().detach().numpy()
        X_E_comparison_cpu = (X + E_comparison).cpu().detach().numpy()
        TN_G_comparison = (TN_G_comparison).cpu().detach().numpy()

    W, H = mynmf.nmf(X, ranks, num_iterations=inner_num_steps, tol=1e-10, device="cuda")
    TN_G_ori = torch.mm(W, H)
    norm2_original_TR_algorithm = torch.norm(X - TN_G_ori).item()
    print(f"Norm2 (Original TR Algorithm): {norm2_original_TR_algorithm}")
    # 记录 norm2 结果到日志
    logging.info(f"Norm2 (Our Algorithm): {norm2_our_algorithm}")
    logging.info(f"Norm2 (Original TR Algorithm): {norm2_original_TR_algorithm}")


    if imshow_enabled:
        # 将张量从 CUDA 移动到 CPU 并转换为 NumPy 数组
        X_cpu = X.cpu().detach().numpy()
        E_cpu = E_optimized.cpu().detach().numpy()
        X_E_cpu = (X + E_optimized).cpu().detach().numpy()
        TN_G = (TN_G).cpu().detach().numpy()
        TN_G_ori = TN_G_ori.cpu().detach().numpy()


        # 创建一个全白的图像
        white_image = np.ones_like(X_cpu)

        # 创建一个全黑的图像
        black_image = np.zeros_like(X_cpu)


        # 获取当前时间并格式化为字符串
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'result/lena_comparison_{current_time}.png'

        images = [
        X_cpu, black_image, X_cpu, TN_G_ori,  # 第一行
        X_cpu, E_cpu, X_E_cpu, TN_G,      # 第二行
        X_cpu, E_comparison_cpu, X_E_comparison_cpu, TN_G_comparison  # 第三行
        ]   

        titles = [
            'Original Image', 'Empty Noise', 'Original Image', 'TR Reconstructed Image',  # 第一行
            'Original Image', 'Ours Adversarial Noise', 'OursAdversarial Noisy Image', 'Ours Reconstructed Image',  # 第二行
            'Original Image', 'ATR Adversarial Noise', 'ATR Adversarial Noisy Image', 'ATR Reconstructed Image'  # 第二行
        ]
    
        display_images(X_cpu, images, titles, 3, 4, save_path=save_path)
        # loss_ours = [loss.cpu().detach().numpy() for loss in loss_ours]
        # loss_ATR = [loss.cpu().detach().numpy() for loss in loss_ATR]

        # plt.figure(figsize=(10, 5))
        # plt.plot(loss_ours, label='Ours')
        # plt.plot(loss_ATR, label='Comparison')
        # plt.xlabel('Step')
        # plt.ylabel('Loss (sqrt)')
        # plt.title('Loss over Steps')
        # plt.legend()
        # plt.grid(True)
        # plt.savefig(f'result/loss_plot_{current_time}.png')
        # plt.show()

        # loss_dict = {
        #     'Ours': loss_ours,
        #     'Comparison': loss_ATR
        # }
        # loss_plot_path = os.path.join( 'result_NMF/loss_plot.png')
        # display_line_graphs(loss_dict, save_path=loss_plot_path)

        # 创建一个包含三个子图的图形
        plt.figure(figsize=(15, 10))

        # 第一个子图：同时显示 loss_ours 和 loss_ATR
        plt.subplot(3, 1, 1)
        plt.plot(loss_ours, label='Ours')
        plt.plot(loss_ATR, label='Comparison')
        plt.xlabel('Step')
        plt.ylabel('Loss (sqrt)')
        plt.title('Loss over Steps')
        plt.legend()
        plt.grid(True)

        # 第二个子图：显示 loss_ours
        plt.subplot(3, 1, 2)
        plt.plot(loss_ours, label='Ours', color='blue')
        plt.xlabel('Step')
        plt.ylabel('Loss (sqrt)')
        plt.title('Loss over Steps (Ours)')
        plt.legend()
        plt.grid(True)

        # 第三个子图：显示 loss_ATR
        plt.subplot(3, 1, 3)
        plt.plot(loss_ATR, label='Comparison', color='orange')
        plt.xlabel('Step')
        plt.ylabel('Loss (sqrt)')
        plt.title('Loss over Steps (Comparison)')
        plt.legend()
        plt.grid(True)

        # 调整子图布局
        plt.tight_layout()

        # 保存图形
        plt.savefig(f'result/loss_plot_{current_time}.png')

        # 显示图形
        plt.show()

        # 保存 loss_ours 和 loss_ATR 到字典并显示折线图
        loss_dict = {
            'Ours': loss_ours,
            'Comparison': loss_ATR
        }
        loss_plot_path = os.path.join('result_NMF/loss_plot.png')
        display_line_graphs(loss_dict, save_path=loss_plot_path)

    