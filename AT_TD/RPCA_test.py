import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorly.decomposition import robust_pca
from datetime import datetime

# 假设输入张量 X
lena = Image.open('TN_G_comp_output.png').convert('RGB')  # 去除 Alpha 通道
# X = np.array(lena)
X = np.array(lena, dtype=np.float32)/255 # 将 PIL 图像对象转换为 NumPy 数组并转换为 float64 类型
# X = torch.tensor(lena/255.0, dtype=torch.float32).to("cuda")

# 调用 robust_pca 进行 TRPCA 分解
L, S = robust_pca(X, reg_E=0.1, n_iter_max=20)

# 创建保存结果的文件夹
output_folder = 'result_RPCA'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取当前时间并格式化为字符串
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# 显示和保存 L
plt.imshow(X)
# plt.title('低秩张量 L')
plt.axis('off')  # 去除坐标轴
plt.savefig(os.path.join(output_folder, f'X_{current_time}.png'), bbox_inches='tight', pad_inches=0)
plt.show()

# 显示和保存 L
plt.imshow(L)
# plt.title('低秩张量 L')
plt.axis('off')  # 去除坐标轴
plt.savefig(os.path.join(output_folder, f'L_{current_time}.png'), bbox_inches='tight', pad_inches=0)
plt.show()

# 显示和保存 S
plt.imshow(S)
# plt.title('稀疏噪声 S')
plt.axis('off')  # 去除坐标轴
plt.savefig(os.path.join(output_folder, f'S_{current_time}.png'), bbox_inches='tight', pad_inches=0)
plt.show()

# print("低秩张量:", L)
# print("稀疏噪声:", S)
