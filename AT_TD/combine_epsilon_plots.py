import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# 创建输出文件夹
output_folder = './AT_TD/combined_epsilon_plots'
os.makedirs(output_folder, exist_ok=True)

# 加载两张图片
img1 = mpimg.imread('./AT_TD/epgilon_compare_results_rank_5_mean_only2/epsilon_vs_loss_mean.png')
img2 = mpimg.imread('./AT_TD/epgilon_compare_results_rank_10_mean_only2/epsilon_vs_loss_mean.png')

# 创建1x2的子图
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# 显示图片
axes[0].imshow(img1)
axes[0].set_title('Rank 5', fontsize=20)
axes[0].axis('off')

axes[1].imshow(img2)
axes[1].set_title('Rank 10', fontsize=20)
axes[1].axis('off')

plt.tight_layout()

# 保存为png和pdf
plt.savefig(os.path.join(output_folder, 'epsilon_vs_loss_combined.png'), format='png', dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_folder, 'epsilon_vs_loss_combined.pdf'), format='pdf', bbox_inches='tight')
plt.close()

print(f"图片已保存到 {output_folder}")
