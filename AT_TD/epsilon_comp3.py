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
import glob


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
imshow_enabled = True
compare_method_enabled = True

# 创建日志文件夹
if log_enabled:
    log_folder = 'logs_AT_TR_plus'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# 配置日志
    log_filename = os.path.join(log_folder, f'AT_TR_compare_{current_time}.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(message)s')

tl.set_backend('pytorch')  # Set tensorly to use PyTorch backend

class OptimizationConfig:
    def __init__(self, epsilon, lr, outer_num_steps, inner_num_steps, outer_dis_num, inner_dis_num, inner_tol, tr_cores):
        self.epsilon = epsilon
        self.lr = lr
        self.outer_num_steps = outer_num_steps
        self.inner_num_steps = inner_num_steps
        self.outer_dis_num = outer_dis_num
        self.inner_dis_num = inner_dis_num
        self.inner_tol = inner_tol
        self.tr_cores = tr_cores

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

def _pad_to_same_len(seq_list, pad_val=np.nan):
    """把不同长度的 step 曲线右侧补 NaN，方便 nanmean/nanstd。"""
    if len(seq_list) == 0:
        return np.empty((0, 0))
    max_len = max(len(s) for s in seq_list)
    out = np.full((len(seq_list), max_len), pad_val, dtype=float)
    for i, s in enumerate(seq_list):
        out[i, :len(s)] = s
    return out

def _nearest_index(arr, val):
    arr = np.asarray(arr, dtype=float)
    return int(np.argmin(np.abs(arr - val)))

def run_multiple_images_and_plot(
    image_dir,
    img_patterns=("*.png",),
    output_folder="./AT_TD/epgilon_compare_results_multi",
    config=None,
    epsilons=None,
    epsilon_for_step_plot=None   # 选一个 epsilon 来画“Loss over Steps”的均值±方差带
):
    os.makedirs(output_folder, exist_ok=True)

    # 收集文件
    image_files = []
    for p in img_patterns:
        image_files.extend(sorted(glob.glob(os.path.join(image_dir, p))))
    if len(image_files) == 0:
        raise RuntimeError("未找到任何图片，请检查路径或通配符。")

    # 结果缓存（最终 loss）
    ours_final_all   = []  # (num_imgs, len(eps))
    oursF_final_all  = []
    ATR_final_all    = []
    TRD_all          = []  # 每张图一个 TRD（与 epsilon 无关）

    # 结果缓存（step 曲线）：按 epsilon 索引存一个 list（里面是每张图的曲线）
    step_curves_ours   = [[] for _ in range(len(epsilons))]
    step_curves_oursF  = [[] for _ in range(len(epsilons))]
    step_curves_ATR    = [[] for _ in range(len(epsilons))]

    for img_path in image_files:
        print(f"[Info] Running on {os.path.basename(img_path)}")
        this_img_ours   = []
        this_img_oursF  = []
        this_img_ATR    = []
        this_img_TRD    = None

        for ei, ep in enumerate(epsilons):
            config.epsilon = float(ep)
            try:
                (t_simpl,
                 ours_last, oursF_last, ATR_last,
                 trd_scalar,
                 ours_curve, oursF_curve, ATR_curve) = process_image(img_path, output_folder, config)

                # 最终 loss（每 epsilon 一个标量）
                this_img_ours.append(ours_last)
                this_img_oursF.append(oursF_last)
                this_img_ATR.append(ATR_last)
                this_img_TRD = trd_scalar  # 取最后一次（相同）

                # step 曲线（每 epsilon 一条曲线，按“图片”收集）
                step_curves_ours[ei].append(np.asarray(ours_curve, dtype=float))
                step_curves_oursF[ei].append(np.asarray(oursF_curve, dtype=float))
                step_curves_ATR[ei].append(np.asarray(ATR_curve, dtype=float))

            except Exception as e:
                print(f"[Warn] {os.path.basename(img_path)} @ epsilon={ep} 失败：{e}")
                # 失败时用 NaN 占位，保证长度对齐
                this_img_ours.append(np.nan)
                this_img_oursF.append(np.nan)
                this_img_ATR.append(np.nan)

        if this_img_TRD is not None:
            TRD_all.append(this_img_TRD)

        ours_final_all.append(this_img_ours)
        oursF_final_all.append(this_img_oursF)
        ATR_final_all.append(this_img_ATR)

    # ---------- 统计：Final Loss vs Epsilon（均值±方差） ----------
    ours_final_all  = np.asarray(ours_final_all, dtype=float)   # (Nimg, Neps)
    oursF_final_all = np.asarray(oursF_final_all, dtype=float)
    ATR_final_all   = np.asarray(ATR_final_all, dtype=float)

    ours_mean  = np.nanmean(ours_final_all,  axis=0)
    ours_std   = np.nanstd(ours_final_all,   axis=0)
    oursF_mean = np.nanmean(oursF_final_all, axis=0)
    oursF_std  = np.nanstd(oursF_final_all,  axis=0)
    ATR_mean   = np.nanmean(ATR_final_all,   axis=0)
    ATR_std    = np.nanstd(ATR_final_all,    axis=0)

    TRD_all = np.asarray(TRD_all, dtype=float)
    TRD_mean = np.nanmean(TRD_all) if TRD_all.size > 0 else np.nan
    TRD_std  = np.nanstd(TRD_all)  if TRD_all.size > 0 else np.nan

    # 画 epsilon 曲线（均值±方差带）
    plt.figure(figsize=(10, 6))
    x = np.asarray(epsilons, dtype=float)

    plt.plot(x, ours_mean,  marker='o', label='Ours')
    plt.fill_between(x, ours_mean - ours_std,  ours_mean + ours_std,  alpha=0.2)

    plt.plot(x, oursF_mean, marker='o', label='Ours_F')
    plt.fill_between(x, oursF_mean - oursF_std, oursF_mean + oursF_std, alpha=0.2)

    plt.plot(x, ATR_mean,   marker='o', label='Comparison')
    plt.fill_between(x, ATR_mean - ATR_std,   ATR_mean + ATR_std,   alpha=0.2)

    # TRD 的均值±方差水平带
    if not np.isnan(TRD_mean):
        plt.axhline(y=TRD_mean, color='r', linestyle='--', label='TRD (mean)')
        plt.fill_between(x, TRD_mean - TRD_std, TRD_mean + TRD_std, alpha=0.1)

    plt.xlabel('Epsilon')
    plt.ylabel('Final Loss')
    plt.title('Final Loss vs Epsilon (Mean ± Std)')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'epsilon_vs_loss_mean_std.png'))
    plt.show()

    # ---------- 统计：Loss over Steps（选定一个 epsilon，做均值±方差曲线） ----------
    if epsilon_for_step_plot is None:
        epsilon_for_step_plot = epsilons[0]  # 默认取第一个
    ei = _nearest_index(np.asarray(epsilons, dtype=float), float(epsilon_for_step_plot))

    # 各方法在该 epsilon 下的 step 曲线，先补齐长度，再 nanmean/nanstd
    ours_mat  = _pad_to_same_len(step_curves_ours[ei])
    oursF_mat = _pad_to_same_len(step_curves_oursF[ei])
    ATR_mat   = _pad_to_same_len(step_curves_ATR[ei])

    if ours_mat.size == 0 and oursF_mat.size == 0 and ATR_mat.size == 0:
        print("[Warn] 选定 epsilon 下没有任何有效的 step 曲线，跳过 step 图。")
        return

    t_max = max(ours_mat.shape[1] if ours_mat.size else 0,
                oursF_mat.shape[1] if oursF_mat.size else 0,
                ATR_mat.shape[1] if ATR_mat.size else 0)
    steps = np.arange(t_max)

    def _mean_std(mat):
        if mat.size == 0:
            return np.full(t_max, np.nan), np.full(t_max, np.nan)
        # 如果矩阵列数不足 t_max，右侧再补 NaN，避免 fill_between 维度错
        if mat.shape[1] < t_max:
            pad = np.full((mat.shape[0], t_max - mat.shape[1]), np.nan)
            mat = np.hstack([mat, pad])
        return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)

    ours_m,  ours_s  = _mean_std(ours_mat)
    oursF_m, oursF_s = _mean_std(oursF_mat)
    ATR_m,   ATR_s   = _mean_std(ATR_mat)

    # 画 step 曲线（均值±方差带）
    plt.figure(figsize=(10, 6))
    plt.plot(steps, ours_m,  label='Ours')
    plt.fill_between(steps, ours_m - ours_s,  ours_m + ours_s,  alpha=0.2)

    plt.plot(steps, oursF_m, label='Ours_F')
    plt.fill_between(steps, oursF_m - oursF_s, oursF_m + oursF_s, alpha=0.2)

    plt.plot(steps, ATR_m,   label='Comparison')
    plt.fill_between(steps, ATR_m - ATR_s,   ATR_m + ATR_s,   alpha=0.2)

    # TRD 水平带（均值±方差）
    if not np.isnan(TRD_mean):
        plt.axhline(y=TRD_mean, color='r', linestyle='--', label='TRD (mean)')
        plt.fill_between(steps, TRD_mean - TRD_std, TRD_mean + TRD_std, alpha=0.1)

    plt.xlabel('Step')
    plt.ylabel('Loss (sqrt)')
    plt.title(f'Loss over Steps @ epsilon={float(epsilons[ei])} (Mean ± Std)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, f'loss_over_steps_mean_std_eps_{float(epsilons[ei])}.png'))
    plt.show()

        # ---------- 保存数据 ----------
    save_path = os.path.join(output_folder, "results_data.npz")
    np.savez(
        save_path,
        epsilons=np.asarray(epsilons, dtype=float),

        ours_final_all=ours_final_all,
        oursF_final_all=oursF_final_all,
        ATR_final_all=ATR_final_all,

        ours_mean=ours_mean, ours_std=ours_std,
        oursF_mean=oursF_mean, oursF_std=oursF_std,
        ATR_mean=ATR_mean, ATR_std=ATR_std,

        TRD_all=TRD_all,
        TRD_mean=TRD_mean,
        TRD_std=TRD_std,

        # step 曲线原始数据（只保存你选定的 epsilon）
        ours_mat=ours_mat,
        oursF_mat=oursF_mat,
        ATR_mat=ATR_mat,
        steps=steps,
        ours_m=ours_m, ours_s=ours_s,
        oursF_m=oursF_m, oursF_s=oursF_s,
        ATR_m=ATR_m, ATR_s=ATR_s,
    )
    print(f"[Info] 数据已保存到 {save_path}")


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
        G_list = tr.tr_decompose2(X+E, G_list_ori, max_iter=config.inner_num_steps, tol=config.inner_tol, dis_num=config.inner_dis_num, device="cuda")
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

def process_image(image_path, output_folder, config):
    image = Image.open(image_path).convert('RGB')
    new_size = (128, 128)
    image_resized = np.array(image.resize(new_size, Image.LANCZOS))
    X = torch.tensor(image_resized / 255.0, dtype=torch.float32).to("cuda")

    base_name = os.path.basename(image_path).split('.')[0]
    output_folder2 = 'Results My picture2'
    if not os.path.exists(output_folder2):
        os.makedirs(output_folder2)

    tr_cores = config.tr_cores

    # 1) 简化版
    start_time = time.time()
    E_optimized_F, TN_G_F, loss_ours_F = optimize_perturbation_simplifed(X, tr_cores, config)
    time_simplified = time.time() - start_time

    # 2) Ours
    X.requires_grad_(True)
    start_time = time.time()
    E_optimized, TN_G, loss_ours = optimize_perturbation(X, tr_cores, config)
    time_optimized = time.time() - start_time
    print(f"optimize_perturbation_simplifed 运行时间: {time_simplified:.4f} 秒")
    print(f"optimize_perturbation 运行时间: {time_optimized:.4f} 秒")

    # 3) 原始 TRD
    X = X.detach()
    tr_cores = [G.detach() for G in tr_cores]
    tr_cores2 = tr.tr_decompose2(X, tr_cores, max_iter=config.inner_num_steps, tol=config.inner_tol, device="cuda")
    TN_G_ori = tl.tr_to_tensor(tr_cores2) 
    norm2_original_TR_algorithm = torch.norm(X - TN_G_ori).item()
    print(f"Norm2 (Original TR Algorithm): {norm2_original_TR_algorithm}")
    logging.info(f"Norm2 (Original TR Algorithm): {norm2_original_TR_algorithm}")

    # 4) Comparison
    E_comparison, TN_G_comparison, loss_ATR = optimize_perturbation_comparison(X, tr_cores, config)

    # 注意：返回值在原来基础上**多返回三条曲线**，便于汇总做均值/方差
    return (
        time_simplified,
        float(loss_ours[-1]),
        float(loss_ours_F[-1]),
        float(loss_ATR[-1]),
        float(norm2_original_TR_algorithm),
        list(map(float, loss_ours)),       # 新增：ours 的 step 曲线
        list(map(float, loss_ours_F)),     # 新增：ours_F 的 step 曲线
        list(map(float, loss_ATR))         # 新增：ATR 的 step 曲线
    )


def run_multiple_images(image_names, input_folder, output_folder, config, epsilons):
    # 存储不同图片在不同 epsilon 下的结果
    all_loss_ours_last = []
    all_loss_ours_F_last = []
    all_loss_ATR_last = []

    for img_name in image_names:
        image_path = os.path.join(input_folder, img_name)
        if not os.path.isfile(image_path):
            print(f"Image {img_name} not found, skipped.")
            continue

        print(f"Running experiments on {img_name}")
        loss_ours_last = []
        loss_ours_F_last = []
        loss_ATR_last = []

        for epsilon in epsilons:
            config.epsilon = epsilon
            time_simplified, loss_ours_val, loss_ours_F_val, loss_ATR_val, norm2_TR = process_image(image_path, output_folder, config)
            loss_ours_last.append(loss_ours_val)
            loss_ours_F_last.append(loss_ours_F_val)
            loss_ATR_last.append(loss_ATR_val)

        all_loss_ours_last.append(loss_ours_last)
        all_loss_ours_F_last.append(loss_ours_F_last)
        all_loss_ATR_last.append(loss_ATR_last)

    # 转换为 numpy 数组方便统计
    all_loss_ours_last = np.array(all_loss_ours_last)
    all_loss_ours_F_last = np.array(all_loss_ours_F_last)
    all_loss_ATR_last = np.array(all_loss_ATR_last)

    # 计算均值和标准差
    ours_mean, ours_std = all_loss_ours_last.mean(axis=0), all_loss_ours_last.std(axis=0)
    ours_F_mean, ours_F_std = all_loss_ours_F_last.mean(axis=0), all_loss_ours_F_last.std(axis=0)
    ATR_mean, ATR_std = all_loss_ATR_last.mean(axis=0), all_loss_ATR_last.std(axis=0)

        # ============ 新的画图部分 (均值 ± 标准差带) ============
    plt.figure(figsize=(10, 6))

    valid_len = ours_mean.shape[0]
    epsilons_valid = epsilons[:valid_len]

    # Ours
    plt.plot(epsilons_valid, ours_mean, marker='o', label='Ours')
    plt.fill_between(epsilons_valid,
                     ours_mean - ours_std,
                     ours_mean + ours_std,
                     alpha=0.2)

    # Ours_F
    plt.plot(epsilons_valid, ours_F_mean, marker='o', label='Ours_F')
    plt.fill_between(epsilons_valid,
                     ours_F_mean - ours_F_std,
                     ours_F_mean + ours_F_std,
                     alpha=0.2)

    # Comparison
    plt.plot(epsilons_valid, ATR_mean, marker='o', label='Comparison')
    plt.fill_between(epsilons_valid,
                     ATR_mean - ATR_std,
                     ATR_mean + ATR_std,
                     alpha=0.2)

    # 红色虚线
    plt.axhline(y=norm2_TR, color='r', linestyle='--', label='TRD')

    plt.xlabel('Epsilon')
    plt.ylabel('Final Loss')
    plt.title('Final Loss vs Epsilon (Mean ± Std)')
    plt.xscale('log')   # 保留 log 横轴
    plt.legend()
    plt.grid(True)

    plt.savefig(f'{output_folder}/loss_plot_epsilon_mean_std.png')
    plt.show()


    return ours_mean, ours_std, ours_F_mean, ours_F_std, ATR_mean, ATR_std

if __name__ == "__main__":
    input_folder = './AT_TD/My picture'
    output_folder = './AT_TD/epgilon_compare_results15'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_folder = './AT_TD/My picture'  # 替换为你的图片文件夹路径
    start = 0 # 从第 10 张图片开始（索引从 0 开始）
    end = 8   # 到第 20 张图片结束（不包括第 20 张）

    # 初始化 image_names_real 列表
    image_names_real = []

    # 获取指定范围的图片文件
    # image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])[start:end]
    # for image_name in image_files:
    #     image_path = os.path.join(image_folder, image_name)
    #     image_names_real.append(image_name) 

    # epsilons = [5 * i for i in range(1, 4)]  # epsilon 列表

    # 初始化 config （这里你沿用之前的设置）
    ranks = [15, 15, 15, 15]
    shape = [128, 128, 3]
    tr_cores = tr.initialize_tr_cores(shape, ranks, device="cuda")
    epsilon_list = [5 * i for i in range(1, 10)]  # 你原来的 epsilons
    config = OptimizationConfig(
        epsilon=epsilon_list[0],
        lr=0.05,
        outer_num_steps=100,
        inner_num_steps=100,
        outer_dis_num=1,
        inner_dis_num=2000,
        inner_tol=1e-10,
        tr_cores=tr_cores
    )
    # 这里可以选择只跑数字文件名：("*.png",) 改成 ("1.png","2.png","3.png")
    run_multiple_images_and_plot(
        image_dir=input_folder,
        img_patterns=sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])[start:end],   # 或者 ("*.png",)
        output_folder=output_folder,
        config=config,
        epsilons=epsilon_list,
        epsilon_for_step_plot=epsilon_list[0]      # 选定一个 epsilon 画 step 曲线
    )
    # run_multiple_images(image_files, image_folder, output_folder, config, epsilons)

