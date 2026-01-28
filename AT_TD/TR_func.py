import torch
import torch.autograd as autograd
import tensorly as tl
from tensorly.tenalg import mode_dot
from tensorly import unfold, fold
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import gc
import copy

def initialize_tr_cores(shape, ranks, device):
    cores = []
    for i in range(len(shape)):
        core_shape = (ranks[i], shape[i], ranks[i + 1])
        core = torch.randn(core_shape, device=device)
        cores.append(core)
    return cores

def show_images_pillow(image1, title1, image2, title2):
    # 转换为 PIL 格式
    img1 = Image.fromarray((image1 * 255).astype(np.uint8))  # 转换为 0-255 的整数值
    img2 = Image.fromarray((image2 * 255).astype(np.uint8))  # 转换为 0-255 的整数值

    # 显示图片
    img1.show(title=title1)
    img2.show(title=title2)


# 将 Tensorly backend 设置为 PyTorch
tl.set_backend('pytorch')

def generalized_k_contraction(X, Y, modes_X, modes_Y, k):
    """
    广义张量 k 收缩积的 GPU 加速实现

    参数:
        X (torch.Tensor): 输入张量 X
        Y (torch.Tensor): 输入张量 Y
        modes_X (list): X 中的模式索引，与 Y 的模式索引对齐
        modes_Y (list): Y 中的模式索引，与 X 的模式索引对齐

    返回:
        torch.Tensor: 收缩后的张量
    """

    # 将张量展开为矩阵
    X_unfolded_T = generalized_k_unfolding(X, modes_X, k).T  # 按第一个模式展开
    Y_unfolded = generalized_k_unfolding(Y, modes_Y, k)  # 按第一个模式展开

    # 对齐的模式进行矩阵乘积
    # contracted = torch.matmul(X_unfolded_T, Y_unfolded)
    contracted = X_unfolded_T @ Y_unfolded

    sorted_dims_X = [X.shape[i] for i in modes_X[k+1:]]
    sorted_dims_Y = [Y.shape[i] for i in modes_Y[k+1:]]
    # 将矩阵重新折叠回张量
    new_shape = sorted_dims_X + sorted_dims_Y
    # result = fold(contracted, mode=0, shape=new_shape)
    result = contracted.reshape(new_shape)

    return result





def generate_sequence(n, N):
    if n > N:
        raise ValueError("n must be less than or equal to N")
    return list(range(n+1, N+1)) + list(range(0, n))

def contract_except_n(cores, n):
    """
    计算除第 n 个核心张量之外的张量积
    """
    N = len(cores)
    # print("N", N)
    
    cont_seq = generate_sequence(n, N-1)
    # print("cont_seq", cont_seq)
  
    full_tensor = cores[cont_seq[0]]
    # print("full_tensor shape", full_tensor.shape)

    for i in range(1, N-1):
        dims = list(range(full_tensor.ndim))
        new_dims = [dims[-1]] + dims[:-1]
        core_i_dims = list(range(cores[cont_seq[i]].ndim))
        full_tensor = generalized_k_contraction(full_tensor, cores[cont_seq[i]], new_dims, core_i_dims, 0)
        if not isinstance(full_tensor, torch.Tensor):
            raise TypeError("full_tensor is not a torch.Tensor")
        # print("full_tensor shape", full_tensor.shape)

    # 将结果转换回 PyTorch 张量
    # result = torch.tensor(full_tensor, requires_grad=True, dtype=torch.float32).to(cores[0].device)
    # result = full_tensor
    return full_tensor

def generalized_k_unfolding(X, modes, k):
    """
    广义张量 k 展开的 GPU 加速实现

    参数:
        X (torch.Tensor): 输入张量
        modes (list): 展开的模式顺序

    返回:
        torch.Tensor: 展开后的矩阵
    """
    # 验证输入参数
    if len(set(modes)) != len(modes):
        raise ValueError("模式索引不能重复！")
    
    if max(modes) >= X.ndim:
        raise ValueError("模式索引超出了张量的维度！")

    # 调整张量模式顺序
    permuted_X = X.permute(*modes)

    # 计算新的矩阵形状
    # print("prod X_k", torch.prod(torch.tensor([X.shape[i] for i in modes[:k]])).item())
    unfold_shape = (torch.prod(torch.tensor(permuted_X.shape[:k+1])),
                    torch.prod(torch.tensor(permuted_X.shape[k+1:])))

    
    # 将调整后的张量 reshape 成矩阵
    unfolded_X = permuted_X.reshape(unfold_shape)

    # del permuted_X, unfold_shape
    # gc.collect()
    # torch.cuda.empty_cache()

    return unfolded_X



def als_update_tr(X, cores, n, device):
    # 确保 X 是一个正确的张量
    # X = tl.tensor(X, requires_grad=True, device=device)
    # norm2X = torch.norm(X)**2
    # print("norm2X", norm2X)
    # 将张量 X 转换为 NumPy 数组
    # X_np = X.cpu().detach().numpy()
    # X_np = np.clip(X_np, 0, 1)  # 将张量值裁剪到 [0, 1] 区间

    # # 显示图像
    # plt.imshow(X_np)
    # plt.title('Image X')
    # plt.colorbar()
    # plt.show()

    # # 保存图像到文件
    # plt.imsave('image_completion_test_X.png', X_np)

    # 计算除 cores[n] 之外的张量积
    X_hat = contract_except_n(cores, n)

    # # 检查 contract_except_n 是否保持了梯度路径
    # print(f"X_hat requires_grad (n={n}):", X_hat.requires_grad)

    dims = list(range(X_hat.dim()))
    new_dims = [dims[-1]] + dims[:-1]
    X_hat_unfold = generalized_k_unfolding(X_hat, new_dims, 1)
    # X_hat_unfold_T = X_hat_unfold.T

    X_dims= list(range(X.dim()))
    new_X_dims = [X_dims[n]] + X_dims[n+1:] + X_dims[:n]
    X_unfold = generalized_k_unfolding(X, new_X_dims, 0)
    # X_unfold = torch.tensor(X_unfold, requires_grad=True, dtype=torch.float32).to(device)
    # X_unfold_T = X_unfold.T

    # # Solve X_hat_unfold^T * core = X_unfold^T in least-squares sense to avoid forming Gram.
    # try:
    #     lstsq_res = torch.linalg.lstsq(X_hat_unfold.transpose(-2, -1), X_unfold.transpose(-2, -1))
    #     solution_t = lstsq_res.solution
    #     new_core = solution_t.transpose(-2, -1)
    # except RuntimeError:
    #     # As a fallback, use pseudo-inverse-based solution
    #     pinv = torch.linalg.pinv(X_hat_unfold.transpose(-2, -1))
    #     new_core = (pinv @ X_unfold.transpose(-2, -1)).transpose(-2, -1)

    # lstsq_res = torch.linalg.lstsq(X_hat_unfold.transpose(-2, -1), X_unfold.transpose(-2, -1))
    # solution_t = lstsq_res.solution
    # new_core = solution_t.transpose(-2, -1)
    # new_core = torch.nan_to_num(new_core)


    pinv_rcond = 1e-6
    try:
        pinv = torch.linalg.pinv(X_hat_unfold, rcond=pinv_rcond)
        new_core = X_unfold @ pinv
    except (RuntimeError, torch.linalg.LinAlgError):
        try:
            lstsq_res = torch.linalg.lstsq(
                X_hat_unfold.transpose(-2, -1),
                X_unfold.transpose(-2, -1),
            )
            new_core = lstsq_res.solution.transpose(-2, -1)
        except (RuntimeError, torch.linalg.LinAlgError):
            gram = X_hat_unfold @ X_hat_unfold.transpose(-2, -1)
            eye = torch.eye(gram.shape[-1], device=device, dtype=gram.dtype)
            damping = 1e-4
            gram = gram + damping * eye
            new_core = (
                X_unfold
                @ X_hat_unfold.transpose(-2, -1)
                @ torch.linalg.solve(gram, eye)
            )

    # new_core = X_unfold @ X_hat_unfold.T @ torch.linalg.pinv(X_hat_unfold @ X_hat_unfold.T)、、、、、、、、、
    # lstsq_res = torch.linalg.lstsq(X_hat_unfold.transpose(-2, -1), X_unfold.transpose(-2, -1))
    # new_core = lstsq_res.solution.transpose(-2, -1)

    # reg = 1e-5
    # new_core = X_unfold@X_hat_unfold.T@torch.linalg.inv(X_hat_unfold@X_hat_unfold.T+ reg * torch.eye(X_hat_unfold.shape[0], device=device))

    new_core = new_core.reshape(cores[n].shape[1], cores[n].shape[0], cores[n].shape[2])
    permuted_new_core = new_core.permute(1,0,2)

    # del new_core, X_hat, X_hat_unfold, X_unfold
    # gc.collect()
    # torch.cuda.empty_cache()
    return permuted_new_core

# Tring to find the aviliable gradient back of the als_update_tr
def als_update_tr_2(X, cores, n, device):

    # 计算除 cores[n] 之外的张量积
    X_hat = contract_except_n(cores, n)

    dims = list(range(X_hat.dim()))
    new_dims = [dims[-1]] + dims[:-1]
    X_hat_unfold = generalized_k_unfolding(X_hat, new_dims, 1)

    X_dims= list(range(X.dim()))
    new_X_dims = [X_dims[n]] + X_dims[n+1:] + X_dims[:n]
    X_unfold = generalized_k_unfolding(X, new_X_dims, 0)
    X_unfold = torch.tensor(X_unfold, requires_grad=True, dtype=torch.float32).to(device)



    new_core = X_unfold@X_hat_unfold.T@torch.linalg.pinv(X_hat_unfold@X_hat_unfold.T)

    new_core = new_core.reshape(cores[n].shape[1], cores[n].shape[0], cores[n].shape[2])
    permuted_new_core = new_core.permute(1,0,2)
    return permuted_new_core

def tr_decompose(X, ranks, max_iter=500, tol=1e-10, dis_num = 10, device="cuda"):
    # 确保 X 是一个正确的张量
    # X = tl.tensor(X, requires_grad=True, device=device)

    shape = list(X.shape)
    cores = initialize_tr_cores(shape, ranks, device)
    for core in cores:
        core.requires_grad_(True)
    prev_loss = float('inf')
    
    for iteration in range(max_iter):
        for n in range(len(shape)):
            cores[n] = als_update_tr(X, cores, n, device)
        
        # 重建张量计算损失
        X_hat = tl.tr_to_tensor(cores)
        loss = torch.norm(X - X_hat) / torch.norm(X)
        
        if abs(prev_loss-loss) < tol:
            # print(f"Converged at iteration {iteration + 1}")
            break

        if iteration != 0 and iteration % dis_num == 0:
            print(f"Iteration {iteration + 1}, Loss: {loss.item():.6f}")
        
        prev_loss = loss

    
    return cores

def tr_decompose2(X, cores_ori, max_iter=500, tol=1e-10, dis_num = 10, device="cuda"):
    # 确保 X 是一个正确的张量
    # X = tl.tensor(X, requires_grad=True, device=device)
    # print("X + E requires_grad:", X.requires_grad)

    shape = list(X.shape)
    prev_loss = float('inf')
    cores = copy.deepcopy(cores_ori)
    
    for iteration in range(max_iter):
        for n in range(len(shape)):
            cores[n] = als_update_tr(X, cores, n, device)
        
        # 重建张量计算损失
        X_hat = tl.tr_to_tensor(cores)
        loss = torch.norm(X - X_hat) / torch.norm(X)
        
        if abs(prev_loss-loss) < tol:
            # print(f"Converged at iteration {iteration + 1}")
            break

        if iteration != 0 and iteration % dis_num == 0:
            print(f"Iteration {iteration + 1}, TRD_Loss: {loss.item():.6f}")
        
        prev_loss = loss
    
    return cores

# Trying to find the aviliable gradient back of the tr_decompose 
def tr_decompose3(X, cores, max_iter=500, tol=1e-10, dis_num = 10, device="cuda"):

    shape = list(X.shape)
    prev_loss = float('inf')
    
    for iteration in range(max_iter):
        for n in range(len(shape)):
            cores[n] = als_update_tr_2(X, cores, n, device)
        
        # 重建张量计算损失
        X_hat = tl.tr_to_tensor(cores)
        loss = torch.norm(X - X_hat) / torch.norm(X)
        
        if abs(prev_loss-loss) < tol:
            print(f"Converged at iteration {iteration + 1}")
            break

        if iteration != 0 and iteration % dis_num == 0:
            print(f"Iteration {iteration + 1}, TRD_Loss: {loss.item():.6f}")
        
        prev_loss = loss
    
    return cores

# def tr_decompose4(X, cores_ori, max_iter=500, tol=1e-10, dis_num = 10, device="cuda"):
#     # 确保 X 是一个正确的张量
#     # X = tl.tensor(X, requires_grad=True, device=device)
#     # print("X + E requires_grad:", X.requires_grad)

#     shape = list(X.shape)
#     prev_loss = float('inf')
#     cores = cores_ori
    
#     for iteration in range(max_iter):
#         for n in range(len(shape)):
#             cores[n] = als_update_tr(X, cores, n, device)
        
#         # 重建张量计算损失
#         X_hat = tl.tr_to_tensor(cores)
#         loss = torch.norm(X - X_hat) / torch.norm(X)
        
#         if abs(prev_loss-loss) < tol:
#             # print(f"Converged at iteration {iteration + 1}")
#             break

#         if iteration != 0 and iteration % dis_num == 0:
#             print(f"Iteration {iteration + 1}, TRD_Loss: {loss.item():.6f}")
        
#         prev_loss = loss
    
#     return cores

def tr_decompose4(X, cores, max_iter=500, tol=1e-10,
                  dis_num=10, device="cuda", track_grad=False):
    prev_loss = torch.tensor(float('inf'), device=device)
    new_cores = list(cores)

    for iteration in range(max_iter):
        for n in range(len(cores)):
            new_cores[n] = als_update_tr(X, new_cores, n, device)
            if torch.isnan(new_cores[n]).any() or torch.isinf(new_cores[n]).any():
                print(f"NaN/Inf in ALS update (core {n}), iteration {iteration + 1}")


        X_hat = tl.tr_to_tensor(new_cores)
        loss = torch.norm(X - X_hat) / torch.norm(X)
        rec_error = torch.norm(X - X_hat)
        if torch.abs(prev_loss - loss) < tol:
            break
        prev_loss = loss
        if iteration != 0 and iteration % dis_num == 0:
            print(f"Iteration {iteration + 1}, TRD_Loss: {loss.item():.10f}, Reconstruction Error: {rec_error.item():.10f}")

        # print(f"Iteration {iteration + 1}, TRD_Loss: {loss.item():.10f}")

    return new_cores


def tr_decompose_gd(
    X,
    cores_ori,
    max_iter=500,
    tol=1e-6,
    lr=1e-3,
    dis_num=10,
    device="cuda",
    normalize_loss=True,
    optimizer_type="adam",
):
    """
    Optimise TR cores via gradient descent (Adam/SGD) instead of ALS updates.

    Args:
        X: Target tensor.
        cores_ori: Initial TR cores (list of tensors).
        max_iter: Number of gradient steps.
        tol: Early-stopping tolerance on loss change.
        lr: Optimiser learning rate.
        dis_num: Print interval.
        device: Device string.
        normalize_loss: If True, loss = ||X-X̂||^2 / ||X||^2 for scale invariance.
        optimizer_type: "adam" or "sgd".

    Returns:
        new_cores: Optimised cores (detached copies).
        history: Dict containing loss / rec_error / grad_norm lists.
    """
    X = X.to(device)
    denom = torch.norm(X)
    eps = 1e-12

    trainable_cores = []
    for core in cores_ori:
        cloned = core.clone().detach().to(device)
        cloned.requires_grad_(True)
        trainable_cores.append(cloned)

    if optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(trainable_cores, lr=lr)
    else:
        optimizer = torch.optim.Adam(trainable_cores, lr=lr)

    prev_loss = float("inf")
    loss_history = []
    rec_history = []
    grad_history = []

    for iteration in range(1, max_iter + 1):
        optimizer.zero_grad()
        X_hat = tl.tr_to_tensor(trainable_cores)
        rec_error = torch.norm(X - X_hat)
        if normalize_loss:
            loss = (rec_error ** 2) / ((denom ** 2) + eps)
        else:
            loss = rec_error ** 2

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            grad_norm = 0.0
            for core in trainable_cores:
                if core.grad is not None:
                    grad_norm += core.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5

            loss_value = float(loss.item())
            rec_value = float(rec_error.item())
            loss_history.append(loss_value)
            rec_history.append(rec_value)
            grad_history.append(grad_norm)

            if abs(prev_loss - loss_value) < tol:
                break
            prev_loss = loss_value

        if iteration % dis_num == 0:
            print(
                f"[TR-GD] Iter {iteration:04d}: loss={loss_value:.6e}, "
                f"rec_error={rec_value:.6e}, grad_norm={grad_norm:.3e}"
            )

    history = {
        "loss": loss_history,
        "rec_error": rec_history,
        "grad_norm": grad_history,
    }
    return [core.detach() for core in trainable_cores], history



def tensor_completion(
    X,
    mask,
    cores_ori,
    max_iter=500,
    tol=1e-10,
    dis_num=10,
    device="cuda",
    missing_init=None,
):
    """
    Perform tensor completion using TR-ALS.
    
    Parameters:
    X: Incomplete tensor with missing values (PyTorch tensor)
    mask: Binary mask indicating observed entries (PyTorch tensor)
    ranks: List of ranks for the TR decomposition
    max_iter: Maximum number of ALS iterations
    tol: Convergence tolerance
    device: Device to perform computations on ("cuda" or "cpu")
    
    missing_init: Optional tensor providing initial values for missing entries.
                  If None, the initial reconstruction from the provided cores is used.

    Returns:
    Completed tensor (PyTorch tensor)
    """
    # Initialize TR cores
    shape = list(X.shape)
    # cores = initialize_tr_cores(shape, ranks, device)
    prev_loss = float("inf")
    mask = mask.to(device=device)
    X = X.to(device)
    mask_bool = mask.bool()
    mask_float = mask.to(dtype=X.dtype)
    inv_mask = (~mask_bool).to(dtype=X.dtype)

    cores = [core.clone().detach().to(device) for core in cores_ori]
    base_completion = tl.tr_to_tensor(cores)

    if missing_init is not None:
        if missing_init.shape != X.shape:
            raise ValueError("missing_init must have the same shape as X.")
        missing_values = missing_init.to(device=device, dtype=X.dtype)
    else:
        missing_values = base_completion

    observed = (X * mask_float).detach()
    working_tensor = observed + missing_values * inv_mask

    for iteration in range(max_iter):
        for n in range(len(shape)):
            cores[n] = als_update_tr(working_tensor, cores, n, device)

        reconstructed = tl.tr_to_tensor(cores)
        masked_diff = observed - reconstructed * mask_float
        denom = torch.norm(observed)
        loss = torch.norm(masked_diff) / (denom + 1e-12)

        working_tensor = reconstructed * inv_mask + observed

        loss_value = loss.item()
        if abs(prev_loss - loss_value) < tol:
            print(f"Converged at iteration {iteration + 1}")
            break

        prev_loss = loss_value

        if iteration != 0 and iteration % dis_num == 0:
            print(f"Iteration {iteration + 1}, Loss: {loss_value}")
    
    # del X_real
    # gc.collect()
    # torch.cuda.empty_cache()
    
    return working_tensor, reconstructed, cores

def reconstruct_tr(cores):
    """
    根据 TR 核心张量重构张量
    """
    N = len(cores)
    X_reconstructed = cores[0]
    for n in range(1, N):
        X_reconstructed = mode_dot(X_reconstructed, cores[n], mode=-1)
    return X_reconstructed
