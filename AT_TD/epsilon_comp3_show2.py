# import numpy as np
# import matplotlib.pyplot as plt
# import os

# # 设置全局字体大小
# plt.rcParams.update({
#     'font.size': 16,  # 设置字体大小
#     'axes.titlesize': 18,  # 图标题字体大小
#     'axes.labelsize': 16,  # 坐标轴标签字体大小
#     'xtick.labelsize': 14,  # x轴刻度字体大小
#     'ytick.labelsize': 14,  # y轴刻度字体大小
#     'legend.fontsize': 14,  # 图例字体大小
# })

# # 加载数据
# data = np.load("./AT_TD/epgilon_compare_results14/results_data.npz", allow_pickle=True)
# output_folder = './AT_TD/epgilon_compare_results_rank_10'
# os.makedirs(output_folder, exist_ok=True)

# epsilons = data["epsilons"]
# ours_mean, ours_std   = data["ours_mean"],  data["ours_std"]
# oursF_mean, oursF_std = data["oursF_mean"], data["oursF_std"]
# ATR_mean, ATR_std     = data["ATR_mean"],   data["ATR_std"]
# TRD_mean, TRD_std     = data["TRD_mean"],   data["TRD_std"]

# # ⚠️ 假设这里保存了每张图片的曲线，里面有样本数
# # 如果没有存 curves，就需要你手动指定 N（图片数量）
# if "ours_curves" in data:
#     N = data["ours_curves"].shape[0]
# else:
#     N = 8  # 如果 npz 里没有曲线数据，这里手动写真实样本数

# # 计算标准误差 SEM
# ours_sem   = ours_std   / np.sqrt(N)
# oursF_sem  = oursF_std  / np.sqrt(N)
# ATR_sem    = ATR_std    / np.sqrt(N)
# TRD_sem    = TRD_std    / np.sqrt(N)

# # -------- Final Loss vs Epsilon --------
# plt.figure(figsize=(10, 6))
# plt.plot(epsilons, ours_mean, marker='o', label='AdaTR')
# plt.fill_between(epsilons, ours_mean - ours_sem, ours_mean + ours_sem, alpha=0.2)

# plt.plot(epsilons, oursF_mean, marker='o', label='FAG-AdaTR')
# plt.fill_between(epsilons, oursF_mean - oursF_sem, oursF_mean + oursF_sem, alpha=0.2)

# plt.plot(epsilons, ATR_mean, marker='o', label='ATTR')
# plt.fill_between(epsilons, ATR_mean - ATR_sem, ATR_mean + ATR_sem, alpha=0.2)

# plt.axhline(y=TRD_mean, color='r', linestyle='--', label='TRD mean')
# plt.fill_between(epsilons, TRD_mean - TRD_sem, TRD_mean + TRD_sem, alpha=0.1)

# plt.xlabel("Epsilon")
# plt.ylabel("$\|\mathcal{X} - \hat{\mathcal{X}}\|$")
# plt.title("Final Loss vs Epsilon (Mean ± SEM)")
# plt.legend()
# plt.grid(True)

# plt.savefig(os.path.join(output_folder, 'epsilon_vs_loss_mean_sem.eps'), format='eps')
# plt.savefig(os.path.join(output_folder, 'epsilon_vs_loss_mean_sem.png'), format='png')
# plt.close()

# # -------- Loss over Steps --------
# steps  = data["steps"]
# ours_m,  ours_s  = data["ours_m"],  data["ours_s"]
# oursF_m, oursF_s = data["oursF_m"], data["oursF_s"]
# ATR_m,   ATR_s   = data["ATR_m"],   data["ATR_s"]

# # 转成 SEM
# ours_sem_curve   = ours_s  / np.sqrt(N)
# oursF_sem_curve  = oursF_s / np.sqrt(N)
# ATR_sem_curve    = ATR_s   / np.sqrt(N)

# plt.figure(figsize=(10, 6))
# plt.plot(steps, ours_m, label="AdaTR")
# plt.fill_between(steps, ours_m - ours_sem_curve, ours_m + ours_sem_curve, alpha=0.2)

# plt.plot(steps, oursF_m, label="FAG-AdaTR")
# plt.fill_between(steps, oursF_m - oursF_sem_curve, oursF_m + oursF_sem_curve, alpha=0.2)

# plt.plot(steps, ATR_m, label="ATTR")
# plt.fill_between(steps, ATR_m - ATR_sem_curve, ATR_m + ATR_sem_curve, alpha=0.2)

# plt.xlabel("Step")
# plt.ylabel("Loss (sqrt)")
# plt.title("Loss over Steps (Mean ± SEM)")
# plt.legend()
# plt.grid(True)

# plt.savefig(os.path.join(output_folder, 'loss_over_steps_mean_sem.eps'), format='eps')
# plt.savefig(os.path.join(output_folder, 'loss_over_steps_mean_sem.png'), format='png')
# plt.close()


# mean only！！！
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置全局字体大小和线条粗细
plt.rcParams.update({
    'font.size': 20,           # 字体大小
    'axes.titlesize': 24,      # 图标题字体大小
    'axes.labelsize': 22,      # 坐标轴标签字体大小
    'xtick.labelsize': 18,     # x轴刻度字体大小
    'ytick.labelsize': 18,     # y轴刻度字体大小
    'legend.fontsize': 18,     # 图例字体大小
    'lines.linewidth': 2.5,    # 线条宽度
    'lines.markersize': 10,    # 标记大小
    'axes.linewidth': 1.5,     # 坐标轴线条宽度
})

# 加载数据
data = np.load("./AT_TD/epgilon_compare_results11/results_data.npz", allow_pickle=True)
output_folder = './AT_TD/epgilon_compare_results_rank_5_mean_only2'
# data = np.load("./AT_TD/epgilon_compare_results14/results_data.npz", allow_pickle=True)
# output_folder = './AT_TD/epgilon_compare_results_rank_10_mean_only'
os.makedirs(output_folder, exist_ok=True)

epsilons = data["epsilons"]
ours_mean   = data["ours_mean"]
oursF_mean  = data["oursF_mean"]
ATR_mean    = data["ATR_mean"]
TRD_mean    = data["TRD_mean"]

# -------- Final Loss vs Epsilon (均值曲线) --------
plt.figure(figsize=(10, 6))
plt.plot(epsilons, ours_mean, marker='o', label='AdaTR')
plt.plot(epsilons, oursF_mean, marker='o', label='FAG-AdaTR')
plt.plot(epsilons, ATR_mean, marker='o', label='ATTR')
plt.axhline(y=TRD_mean, color='r', linestyle='--', linewidth=2.5, label='TRD mean')

plt.xlabel("Epsilon")
plt.ylabel("$\|\mathcal{X} - \hat{\mathcal{X}}\|_F^2$")
# plt.title("Final Loss vs Epsilon (Mean)")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(output_folder, 'epsilon_vs_loss_mean.eps'), format='eps')
plt.savefig(os.path.join(output_folder, 'epsilon_vs_loss_mean.png'), format='png')
plt.savefig(os.path.join(output_folder, 'epsilon_vs_loss_mean.pdf'), format='pdf')
plt.close()

# -------- Loss over Steps (均值曲线) --------
steps  = data["steps"]
ours_m   = data["ours_m"]
oursF_m  = data["oursF_m"]
ATR_m    = data["ATR_m"]

plt.figure(figsize=(10, 6))
plt.plot(steps, ours_m, label="AdaTR")
plt.plot(steps, oursF_m, label="FAG-AdaTR")
# plt.plot(steps, ATR_m, label="ATTR")

plt.xlabel("Step")
plt.ylabel("$\|\mathcal{X} - \hat{\mathcal{X}}\|_F^2$")
# plt.title("Loss over Steps (Mean)")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(output_folder, 'loss_over_steps_mean.eps'), format='eps')
plt.savefig(os.path.join(output_folder, 'loss_over_steps_mean.png'), format='png')
plt.savefig(os.path.join(output_folder, 'loss_over_steps_mean.pdf'), format='pdf')
plt.close()
