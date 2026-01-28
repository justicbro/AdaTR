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


# 定义一个布尔变量来控制日志记录
log_enabled = False

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



def objective_function_ATTR(X, G_list, E, inner_num_steps = 100, inner_tol = 1e-10):
    """
    Compute the objective function ||X - TN([G^(k)])||_F^2.
    X: Original tensor (PyTorch tensor)
    G_list: List of factor tensors in tensor train format
    E: Perturbation tensor (PyTorch tensor)
    """
    EC = E.clone().detach()
    # X_tube = X + EC
    # random_noise = torch.randn_like(X, device='cuda') * 0.1
    # X_tube = X + random_noise

    # tr_cores = tr.tr_decompose3(X_tube, G_list, max_iter=inner_num_steps, tol=inner_tol, dis_num = 2000, device="cuda")
    tr_cores = tr.tr_decompose2(X+EC, G_list, max_iter=inner_num_steps, tol=inner_tol, dis_num = 2000, device="cuda")

    # # 确保 tr_cores 启用了梯度追踪
    # for core in tr_cores:
    #     core.requires_grad_(True)

    # # 释放不再需要的变量
    # del EC, X_tube
    # torch.cuda.empty_cache()

    # E = E.permute(1, 0, *range(2, E.dim()))
    # E = E.permute(1, 0, *range(2, E.dim()))
    # tr_cores[0] = E
    # tr_cores[0] = tr.als_update_tr(X_tube, tr_cores, 0, device='cuda')
    # tr_cores[0] = torch.tensor(tr_cores[0], requires_grad=True, dtype=torch.float32).to('cuda')

    TN_G = tl.tr_to_tensor(tr_cores)  # Reconstruct tensor from tensor train
   
    # E = tl.tensor(E, requires_grad=True, device='cuda')
    return -torch.norm(X + E - TN_G) ** 2, TN_G



def optimize_perturbation(X, G_list, config: OptimizationConfig):
    """
    Optimize perturbation tensor \mathcal{E} using gradient descent.
    X: Original tensor (PyTorch tensor)
    G_list: List of factor tensors in tensor train format
    epsilon: Frobenius norm constraint for \mathcal{E}
    lr: Learning rate for gradient descent
    num_steps: Number of gradient descent steps
    """
    # Initialize perturbation tensor \mathcal{E}
    # E = torch.randn_like(G_list[0], requires_grad=True, device='cuda')
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

        for G in G_list:
            G.requires_grad_(True)
        X.requires_grad_(True)

        # Compute loss
        loss, TN_G = objective_function_ATTR(X, G_list, E, config.inner_num_steps, config.inner_tol)

        # grads = torch.autograd.grad(loss, E, retain_graph=True, allow_unused=True)
        # print(f"Gradient of E: {grads}")
        # Backpropagation
        loss.backward(retain_graph=True)

        # # 检查梯度
        # print("梯度在反向传播后的值：", E.grad)

        # Update E
        optimizer.step()

        # Apply the constraint ||E||_F^2 <= epsilon
        scaling_factor = (config.epsilon ** 0.5) / torch.norm(E)
        if torch.norm(E) ** 2 > config.epsilon:
            E.data = scaling_factor * E

        rse_E = torch.norm(E - E_old)
        print(f"Step {step}, RSE_E: {rse_E}")

        # 记录 norm2 结果到日志
        loss_sqrt = math.sqrt(abs(loss))
        loss_values.append(loss_sqrt)
        logging.info(f"Step {step} norm(X - TN_G_Ours): {loss_sqrt}")

        if step % config.outer_dis_num == 0:
            print(f"Step {step}, Loss: {loss.item()}")
            
            

    return E, TN_G, loss_values


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
    epsilon = (1e2)**2 # Frobenius norm constraint for perturbation tensor
    logging.info(f"Norm2 (epsilon): {epsilon}")
    inner_tol = 1e-10 # Tolerance for inner optimization
    lr = 0.01  # Learning rate for gradient descent
    outer_num_steps = 20 # Number of outer optimization steps
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

    # Optimize perturbation \mathcal{E}
    X.requires_grad_(True)
    E_optimized, TN_G, loss_ours = optimize_perturbation(X, tr_cores, config)

    # 计算 norm2
    norm2_our_algorithm = torch.norm(X - TN_G).item()

    tr_cores = tr.tr_decompose2(X, tr_cores, max_iter=inner_num_steps, tol=1e-10, device="cuda")
    TN_G_ori = tl.tr_to_tensor(tr_cores) 
    norm2_original_TR_algorithm = torch.norm(X - TN_G_ori).item()


    print(f"Norm2 (Original TR Algorithm): {norm2_original_TR_algorithm}")
    logging.info(f"Norm2 (Original TR Algorithm): {norm2_original_TR_algorithm}")

    # 打印 norm2 结果
    print(f"Norm2 (Our Algorithm): {norm2_our_algorithm}")
    # 记录 norm2 结果到日志
    logging.info(f"Norm2 (Our Algorithm): {norm2_our_algorithm}")


