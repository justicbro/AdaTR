import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

output_folder = './AT_TD/epgilon_compare_results_rank_sum'
os.makedirs(output_folder, exist_ok=True)

# 创建一个 1x2 的子图
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 左边放 rank10
img1 = plt.imread("./AT_TD/epgilon_compare_results_rank_5_mean_only/epsilon_vs_loss_mean.eps")
axes[0].imshow(img1)
axes[0].axis("off")
axes[0].set_title("Rank = 10")

# 右边放 rank15
img2 = plt.imread("./AT_TD/epgilon_compare_results_rank_10_mean_only/epsilon_vs_loss_mean.eps")
axes[1].imshow(img2)
axes[1].axis("off")
axes[1].set_title("Rank = 15")

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "compare.pdf"), format="pdf")  # 矢量 PDF
plt.savefig(os.path.join(output_folder, "compare.eps"), format="eps")  # 矢量 EPS
