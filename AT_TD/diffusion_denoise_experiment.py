"""
Diffusion去噪实验：验证diffusion是否可以去除对抗噪声（AdaTR, FAG-AdaTR, ATTR）

这个文件独立运行，不会修改主文件的功能。

使用方法：
    1. 使用预训练diffusion模型（推荐）:
       pip install diffusers transformers accelerate
       python diffusion_denoise_experiment.py \
           --input-folder "./AT_TD/My picture" \
           --start-index 1 --end-index 1 \
           --use-pretrained \
           --diffusion-model "google/ddpm-celebahq-256"
    
    2. 使用简单未训练网络（baseline）:
       python diffusion_denoise_experiment.py \
           --input-folder "./AT_TD/My picture" \
           --start-index 1 --end-index 1

关于预训练模型：
    可用的预训练diffusion模型（需要先安装diffusers库）:
    - google/ddpm-celebahq-256: 人脸图像，256x256
    - google/ddpm-church-256: 教堂图像，256x256
    - google/ddpm-bedroom-256: 卧室图像，256x256
    
    安装命令: pip install diffusers transformers accelerate
    
    注意：某些模型可能需要特定的图像尺寸，代码会自动resize到256x256。
"""

# ==================== 关键：必须在导入torch之前设置CUDA_VISIBLE_DEVICES ====================
# 一旦torch被导入并初始化CUDA，CUDA_VISIBLE_DEVICES就不能再改变了
import os
import sys

# 在导入任何可能使用CUDA的库之前，先解析--gpu参数并设置CUDA_VISIBLE_DEVICES
_gpu_arg = None
if '--gpu' in sys.argv:
    gpu_idx = sys.argv.index('--gpu')
    if gpu_idx + 1 < len(sys.argv):
        try:
            _gpu_arg = int(sys.argv[gpu_idx + 1])
            # 必须在导入torch之前设置CUDA_VISIBLE_DEVICES
            os.environ["CUDA_VISIBLE_DEVICES"] = str(_gpu_arg)
            print(f"✓ 在导入torch之前设置GPU: {_gpu_arg} (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")
        except (ValueError, IndexError):
            pass

# 现在可以安全地导入torch和其他库了
import argparse
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import scipy.io

# 尝试导入diffusers库（用于预训练diffusion模型）
try:
    from diffusers import DDIMScheduler, UNet2DModel, DDPMScheduler
    from diffusers import StableDiffusionPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False
    print("警告: diffusers库未安装。安装命令: pip install diffusers transformers accelerate")

# 尝试导入torchvision的预训练模型
try:
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

# 记录导入AT_AdaTR2之前的CUDA_VISIBLE_DEVICES设置（如果有）
_cuda_visible_devices_before_import = os.environ.get("CUDA_VISIBLE_DEVICES", None)

from AT_AdaTR2 import (
    OptimizationConfig,
    clone_config,
    scaled_config_for_tensor,
    initialize_tr_cores_svd,
    clone_tr_cores,
    load_video_tensor,
    calculate_psnr,
)
from attack_methods import (
    optimize_perturbation,  # AdaTR
    optimize_perturbation_simplifed,  # FAG-AdaTR
    optimize_perturbation_comparison,  # ATTR
)

# 设备将在main函数中根据GPU设置动态确定
# 这里先设置一个默认值，但会在main函数中更新
def get_device():
    """获取当前设备（在GPU设置完成后调用）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using device: {device}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print(f"Using device: {device}")
    return device

# 临时设置device（会在main函数中更新）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== Diffusion模型实现 ====================

class SimpleUNet(nn.Module):
    """简单的U-Net用于diffusion去噪"""
    
    def __init__(self, in_channels=3, out_channels=3, base_channels=64):
        super().__init__()
        
        # 编码器
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        
        # 瓶颈
        self.bottleneck = self._conv_block(base_channels * 4, base_channels * 8)
        
        # 解码器
        self.dec3 = self._conv_block(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.dec2 = self._conv_block(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec1 = self._conv_block(base_channels * 2 + base_channels, base_channels)
        
        # 输出层
        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        
        # 瓶颈
        b = self.bottleneck(p3)
        
        # 解码（处理尺寸不匹配的情况）
        d3 = self.upsample(b)
        # 如果尺寸不匹配，进行crop或padding
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upsample(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upsample(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.final(d1)
        return out


class PretrainedDiffusionDenoiser:
    """
    使用预训练的Diffusion模型进行去噪
    
    注意：标准的DDPM模型（如google/ddpm-*）是用于**生成**图像的，不是专门用于去噪的。
    它们从纯噪声生成图像，而不是从带噪声的图像恢复干净图像。
    对于对抗噪声去噪，效果可能不理想。
    
    更好的选择：
    1. 使用专门训练的去噪模型（如DnCNN, REDNet等）
    2. 使用图像恢复/超分辨率模型
    3. 自己训练去噪网络
    """
    
    def __init__(
        self, 
        model_name: str = "google/ddpm-celebahq-256", 
        num_inference_steps: int = 50,
        noise_level: float = 0.3,
        start_step_ratio: float = 0.3,
        guidance_scale: float = 1.0,
    ):
        """
        Args:
            model_name: 预训练模型名称
                - "google/ddpm-celebahq-256": Google的DDPM模型（人脸，256x256）- 用于生成，不是去噪
                - "google/ddpm-church-256": Google的DDPM模型（教堂，256x256）- 用于生成，不是去噪
                - "google/ddpm-bedroom-256": Google的DDPM模型（卧室，256x256）- 用于生成，不是去噪
                - 或其他diffusers支持的模型
            num_inference_steps: 推理步数（更多步数通常效果更好但更慢）
            noise_level: 添加的噪声水平（0.0-1.0），用于模拟diffusion过程
            start_step_ratio: 从哪个时间步开始去噪（0.0-1.0），0.0表示从纯噪声开始，1.0表示几乎不去噪
            guidance_scale: 引导强度（通常1.0，可以尝试1.0-3.0）
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("需要安装diffusers库: pip install diffusers transformers accelerate")
        
        self.num_inference_steps = num_inference_steps
        self.model_name = model_name
        self.noise_level = noise_level
        self.start_step_ratio = start_step_ratio
        self.guidance_scale = guidance_scale
        
        try:
            # 尝试加载预训练的UNet模型和scheduler
            print(f"正在加载预训练模型: {model_name}...")
            self.unet = UNet2DModel.from_pretrained(model_name).to(device)
            
            # 尝试加载scheduler，如果失败则使用默认的DDIMScheduler
            try:
                self.scheduler = DDIMScheduler.from_pretrained(model_name)
            except:
                print("无法加载模型特定的scheduler，使用默认DDIMScheduler")
                self.scheduler = DDIMScheduler(
                    num_train_timesteps=1000,
                    beta_start=0.0001,
                    beta_end=0.02,
                    beta_schedule="linear",
                    clip_sample=True,
                )
            
            print(f"成功加载预训练模型: {model_name}")
        except Exception as e:
            print(f"加载模型 {model_name} 失败: {e}")
            print("\n可用的模型选项:")
            print("  - google/ddpm-celebahq-256")
            print("  - google/ddpm-church-256")
            print("  - google/ddpm-bedroom-256")
            raise
    
    def denoise(self, noisy_image: torch.Tensor) -> torch.Tensor:
        """
        使用预训练diffusion模型去噪
        
        Args:
            noisy_image: 带噪声的图像 [C, H, W] 或 [B, C, H, W]
        
        Returns:
            去噪后的图像
        """
        if self.unet is None:
            raise ValueError("模型未正确加载")
        
        if noisy_image.dim() == 3:
            noisy_image = noisy_image.unsqueeze(0)
        
        # 检查并调整图像尺寸（某些模型需要特定尺寸）
        B, C, H, W = noisy_image.shape
        
        # 先检查图像范围，决定padding值
        # 如果图像在[0, 1]范围，padding用0；如果在[-1, 1]范围，padding用-1
        image_min = noisy_image.min().item()
        image_max = noisy_image.max().item()
        is_normalized = image_max <= 1.0 and image_min >= 0.0
        padding_value = 0.0 if is_normalized else -1.0
        
        # DDPM模型需要正方形输入（通常是256x256）
        # 如果图像已经是正方形且尺寸合适，就不需要resize
        target_size = 256
        if H != target_size or W != target_size:
            # 如果图像不是正方形，先resize到正方形（保持宽高比，然后padding）
            if H != W:
                # 计算缩放比例，使最大边等于target_size
                scale = target_size / max(H, W)
                new_H = max(1, int(round(H * scale)))
                new_W = max(1, int(round(W * scale)))
                noisy_image = F.interpolate(
                    noisy_image,
                    size=(new_H, new_W),
                    mode='bilinear',
                    align_corners=False
                )
                # Padding到正方形（居中padding）
                pad_h = (target_size - new_H) // 2
                pad_w = (target_size - new_W) // 2
                noisy_image = F.pad(
                    noisy_image,
                    (pad_w, target_size - new_W - pad_w, pad_h, target_size - new_H - pad_h),
                    mode='constant',
                    value=padding_value
                )
            else:
                # 已经是正方形，直接resize到target_size
                noisy_image = F.interpolate(
                    noisy_image,
                    size=(target_size, target_size),
                    mode='bilinear',
                    align_corners=False
                )
            need_resize_back = True
            orig_size = (H, W)
        else:
            need_resize_back = False
        
        # 确保图像在[-1, 1]范围（diffusers通常使用这个范围）
        if noisy_image.max() <= 1.0:
            noisy_image = noisy_image * 2.0 - 1.0
        
        self.unet.eval()
        self.scheduler.set_timesteps(self.num_inference_steps)
        
        with torch.no_grad():
            # 重要：标准DDPM模型是从纯噪声生成图像，不是从带噪声图像去噪
            # 我们需要使用一种更合适的方法来利用diffusion模型进行去噪
            
            # 方法：使用DDIM的inpainting/denoising策略
            # 1. 将带噪声图像视为"部分去噪"的中间状态
            # 2. 估计对应的噪声水平（timestep）
            # 3. 从该timestep开始去噪
            
            # 计算噪声水平：通过比较原始图像和带噪声图像的差异来估计
            # 这里我们使用一个简化的方法：根据noise_level参数估计起始timestep
            start_step = int(self.num_inference_steps * (1.0 - self.noise_level))
            start_step = max(0, min(start_step, self.num_inference_steps - 1))
            
            # 设置timesteps
            self.scheduler.set_timesteps(self.num_inference_steps)
            timesteps = self.scheduler.timesteps[start_step:]
            
            # 关键改进：正确地将带噪声图像编码到diffusion的噪声空间
            # 在DDPM中，x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
            # 我们需要将带噪声图像（x_noisy）编码到x_t
            
            # 获取起始timestep
            start_timestep = self.scheduler.timesteps[start_step] if start_step < len(self.scheduler.timesteps) else self.scheduler.timesteps[-1]
            
            # 计算alpha_bar（累积的alpha值）
            alpha_bar = self.scheduler.alphas_cumprod[start_timestep] if hasattr(self.scheduler, 'alphas_cumprod') else 0.5
            
            # 将带噪声图像编码到x_t
            # 方法：根据diffusion的前向过程，添加适当的噪声
            # x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * epsilon
            # 我们假设带噪声图像是x_0（虽然它已经有噪声），然后添加更多噪声使其成为x_t
            sqrt_alpha_bar = alpha_bar ** 0.5
            sqrt_one_minus_alpha_bar = (1.0 - alpha_bar) ** 0.5
            
            # 添加噪声，将图像编码到对应的timestep
            noise = torch.randn_like(noisy_image)
            latents = sqrt_alpha_bar * noisy_image + sqrt_one_minus_alpha_bar * noise
            latents = torch.clamp(latents, -1.0, 1.0)
            
            # DDIM去噪过程
            for i, t in enumerate(timesteps):
                # 预测噪声
                t_tensor = t.unsqueeze(0).repeat(latents.shape[0]).to(device)
                noise_pred = self.unet(latents, t_tensor).sample
                
                # 使用scheduler更新latents
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                
                # 在去噪过程中，可以添加一些约束来保持图像结构
                latents = torch.clamp(latents, -1.0, 1.0)
        
        # 转换回[0, 1]范围
        denoised = (latents + 1.0) / 2.0
        denoised = torch.clamp(denoised, 0.0, 1.0)
        
        # 如果需要，resize回原始尺寸
        if need_resize_back:
            denoised = F.interpolate(
                denoised,
                size=orig_size,
                mode='bilinear',
                align_corners=False
            )
        
        if denoised.shape[0] == 1:
            denoised = denoised.squeeze(0)
        
        return denoised


class PretrainedDenoisingModel:
    """使用其他预训练的去噪模型（如DnCNN等）"""
    
    def __init__(self, model_type: str = "dncnn"):
        """
        Args:
            model_type: 模型类型
                - "dncnn": DnCNN去噪网络（需要下载预训练权重）
                - "unet": 预训练的U-Net去噪模型
        """
        self.model_type = model_type
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载预训练模型"""
        if self.model_type == "dncnn":
            # DnCNN模型结构
            self.model = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                *[nn.Sequential(
                    nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True)
                ) for _ in range(15)],
                nn.Conv2d(64, 3, kernel_size=3, padding=1, bias=False)
            ).to(device)
            
            # 尝试加载预训练权重（如果有的话）
            # 注意：这里需要用户提供预训练权重路径
            print("警告: DnCNN模型需要预训练权重。当前使用随机初始化。")
            print("可以从以下链接下载预训练权重:")
            print("https://github.com/cszn/DnCNN")
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        self.model.eval()
    
    def denoise(self, noisy_image: torch.Tensor) -> torch.Tensor:
        """去噪"""
        if noisy_image.dim() == 3:
            noisy_image = noisy_image.unsqueeze(0)
        
        with torch.no_grad():
            # 预测噪声
            predicted_noise = self.model(noisy_image)
            # 减去预测的噪声
            denoised = noisy_image - predicted_noise
            denoised = torch.clamp(denoised, 0.0, 1.0)
        
        if denoised.shape[0] == 1:
            denoised = denoised.squeeze(0)
        
        return denoised


class SimpleDiffusionDenoiser:
    """
    简单的Diffusion去噪器（未训练的baseline）
    
    注意：这是一个简化的实现。对于真正的diffusion去噪，请使用PretrainedDiffusionDenoiser。
    """
    
    def __init__(self, model: Optional[nn.Module] = None, use_iterative: bool = False, num_iterations: int = 5):
        """
        Args:
            model: 去噪模型，如果为None则使用简单的UNet
            use_iterative: 是否使用迭代去噪（多次应用模型）
            num_iterations: 迭代次数
        """
        if model is None:
            self.model = SimpleUNet().to(device)
        else:
            self.model = model.to(device)
        
        self.model.eval()
        self.use_iterative = use_iterative
        self.num_iterations = num_iterations
    
    def denoise(self, noisy_image: torch.Tensor) -> torch.Tensor:
        """
        使用模型去噪
        
        Args:
            noisy_image: 带噪声的图像 [C, H, W] 或 [B, C, H, W]
        
        Returns:
            去噪后的图像
        """
        if noisy_image.dim() == 3:
            noisy_image = noisy_image.unsqueeze(0)
        
        with torch.no_grad():
            if self.use_iterative:
                # 迭代去噪：多次应用模型
                x = noisy_image.clone()
                for _ in range(self.num_iterations):
                    residual = self.model(x)
                    x = x - 0.1 * residual  # 减去预测的噪声
                    x = torch.clamp(x, 0.0, 1.0)
                denoised = x
            else:
                # 直接去噪：模型直接预测干净图像
                denoised = self.model(noisy_image)
                denoised = torch.clamp(denoised, 0.0, 1.0)
        
        if denoised.shape[0] == 1:
            denoised = denoised.squeeze(0)
        
        return denoised


class SimpleGaussianDenoiser:
    """简单的高斯去噪器（作为baseline）"""
    
    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma
    
    def denoise(self, noisy_image: torch.Tensor) -> torch.Tensor:
        """使用高斯滤波去噪"""
        if noisy_image.dim() == 3:
            noisy_image = noisy_image.unsqueeze(0)
        
        # 转换为numpy进行高斯滤波
        img_np = noisy_image.cpu().numpy()
        if img_np.ndim == 4:
            img_np = img_np[0]
        
        from scipy import ndimage
        denoised_np = ndimage.gaussian_filter(img_np, sigma=self.sigma)
        denoised = torch.from_numpy(denoised_np).float().to(noisy_image.device)
        
        if noisy_image.shape[0] == 1:
            denoised = denoised.unsqueeze(0)
        
        return torch.clamp(denoised, 0.0, 1.0)


# ==================== 对抗噪声保存/加载 ====================

def get_adversarial_noise_cache_path(output_folder: str, base_name: str, attack_name: str, epsilon: float) -> str:
    """获取对抗噪声缓存文件路径"""
    cache_dir = os.path.join(output_folder, "adversarial_noise_cache")
    os.makedirs(cache_dir, exist_ok=True)
    # 文件名包含base_name、attack_name和epsilon，便于识别
    cache_filename = f"{base_name}_{attack_name}_eps{epsilon:.6f}.pt"
    return os.path.join(cache_dir, cache_filename)


def save_adversarial_noise(
    E: torch.Tensor,
    X_adv: torch.Tensor,
    output_folder: str,
    base_name: str,
    attack_name: str,
    epsilon: float,
    metadata: Optional[Dict] = None,
):
    """保存对抗噪声结果"""
    cache_path = get_adversarial_noise_cache_path(output_folder, base_name, attack_name, epsilon)
    save_data = {
        'E': E.detach().cpu(),  # 扰动
        'X_adv': X_adv.detach().cpu(),  # 对抗图像
        'epsilon': epsilon,
        'attack_name': attack_name,
        'base_name': base_name,
        'metadata': metadata or {},
    }
    torch.save(save_data, cache_path)
    print(f"  已保存对抗噪声到: {cache_path}")


def load_adversarial_noise(
    output_folder: str,
    base_name: str,
    attack_name: str,
    epsilon: float,
    device: torch.device,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """加载已保存的对抗噪声结果"""
    cache_path = get_adversarial_noise_cache_path(output_folder, base_name, attack_name, epsilon)
    if os.path.exists(cache_path):
        try:
            save_data = torch.load(cache_path, map_location='cpu')
            E = save_data['E'].to(device)
            X_adv = save_data['X_adv'].to(device)
            print(f"  ✓ 从缓存加载对抗噪声: {cache_path}")
            print(f"    扰动范数: {torch.norm(E).item():.6f}, epsilon: {save_data.get('epsilon', 'unknown')}")
            return (E, X_adv)
        except Exception as e:
            print(f"  ⚠️  加载缓存失败: {e}，将重新生成")
            return None
    return None


# ==================== 对抗噪声生成 ====================

def generate_adversarial_perturbations(
    X: torch.Tensor,
    tr_cores_init: List[torch.Tensor],
    config: OptimizationConfig,
    output_folder: Optional[str] = None,
    base_name: Optional[str] = None,
    force_regenerate: bool = False,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    生成三种对抗噪声：AdaTR, FAG-AdaTR, ATTR
    
    Args:
        X: 输入图像 [H, W, C]
        tr_cores_init: TR核心初始化
        config: 优化配置
        output_folder: 输出文件夹（用于保存/加载缓存）
        base_name: 基础名称（用于保存/加载缓存）
        force_regenerate: 是否强制重新生成（忽略缓存）
    
    Returns:
        Dict with keys: 'adatr', 'fag_adatr', 'attr'
        Each value is (perturbation, attacked_tensor)
    """
    results = {}
    epsilon = config.epsilon
    
    # AdaTR
    print("生成AdaTR对抗噪声...")
    if not force_regenerate and output_folder and base_name:
        cached = load_adversarial_noise(output_folder, base_name, 'adatr', epsilon, device)
        if cached is not None:
            results['adatr'] = cached
        else:
            try:
                E_adatr, _, _ = optimize_perturbation(
                    X.clone().detach(),
                    clone_tr_cores(tr_cores_init),
                    config,
                )
                X_adatr = X + E_adatr
                results['adatr'] = (E_adatr, X_adatr)
                print(f"AdaTR扰动范数: {torch.norm(E_adatr).item():.6f}")
                # 保存结果
                if output_folder and base_name:
                    save_adversarial_noise(E_adatr, X_adatr, output_folder, base_name, 'adatr', epsilon)
            except Exception as e:
                print(f"AdaTR生成失败: {e}")
                results['adatr'] = None
    else:
        try:
            E_adatr, _, _ = optimize_perturbation(
                X.clone().detach(),
                clone_tr_cores(tr_cores_init),
                config,
            )
            X_adatr = X + E_adatr
            results['adatr'] = (E_adatr, X_adatr)
            print(f"AdaTR扰动范数: {torch.norm(E_adatr).item():.6f}")
            # 保存结果
            if output_folder and base_name:
                save_adversarial_noise(E_adatr, X_adatr, output_folder, base_name, 'adatr', epsilon)
        except Exception as e:
            print(f"AdaTR生成失败: {e}")
            results['adatr'] = None
    
    # FAG-AdaTR
    print("生成FAG-AdaTR对抗噪声...")
    if not force_regenerate and output_folder and base_name:
        cached = load_adversarial_noise(output_folder, base_name, 'fag_adatr', epsilon, device)
        if cached is not None:
            results['fag_adatr'] = cached
        else:
            try:
                # 检查输入X的格式和统计信息
                print(f"  输入X的形状: {X.shape}, 格式: [H, W, C]")
                print(f"  输入X的统计: 均值={X.mean().item():.6f}, 标准差={X.std().item():.6f}")
                print(f"  配置参数: epsilon={config.epsilon}, outer_num_steps_attack={config.outer_num_steps_attack}")
                
                # 调用FAG-AdaTR优化
                E_fag, TN_fag, loss_list = optimize_perturbation_simplifed(
                    X.clone().detach(),
                    clone_tr_cores(tr_cores_init),
                    config,
                )
                # 确保E_fag是detached的
                E_fag = E_fag.detach().clone()
                X_fag = X + E_fag
                
                # 验证X_fag是否正确
                print(f"  FAG-AdaTR扰动范数: {torch.norm(E_fag).item():.6f}")
                print(f"  FAG-AdaTR扰动统计: 均值={E_fag.mean().item():.6f}, "
                      f"标准差={E_fag.std().item():.6f}, "
                      f"最大值={E_fag.max().item():.6f}, 最小值={E_fag.min().item():.6f}")
                
                # 检查优化是否执行了（通过loss_list）
                if loss_list and len(loss_list) > 0:
                    print(f"  优化执行了 {len(loss_list)} 步")
                    print(f"  初始loss: {loss_list[0]:.6f}, 最终loss: {loss_list[-1]:.6f}")
                
                results['fag_adatr'] = (E_fag, X_fag)
                # 保存结果
                if output_folder and base_name:
                    save_adversarial_noise(E_fag, X_fag, output_folder, base_name, 'fag_adatr', epsilon)
            except Exception as e:
                print(f"FAG-AdaTR生成失败: {e}")
                import traceback
                traceback.print_exc()
                results['fag_adatr'] = None
    else:
        try:
            # 检查输入X的格式和统计信息
            print(f"  输入X的形状: {X.shape}, 格式: [H, W, C]")
            print(f"  输入X的统计: 均值={X.mean().item():.6f}, 标准差={X.std().item():.6f}")
            print(f"  配置参数: epsilon={config.epsilon}, outer_num_steps_attack={config.outer_num_steps_attack}")
            
            # 调用FAG-AdaTR优化
            E_fag, TN_fag, loss_list = optimize_perturbation_simplifed(
                X.clone().detach(),
                clone_tr_cores(tr_cores_init),
                config,
            )
            # 确保E_fag是detached的
            E_fag = E_fag.detach().clone()
            X_fag = X + E_fag
            
            # 验证X_fag是否正确
            print(f"  FAG-AdaTR扰动范数: {torch.norm(E_fag).item():.6f}")
            print(f"  FAG-AdaTR扰动统计: 均值={E_fag.mean().item():.6f}, "
                  f"标准差={E_fag.std().item():.6f}, "
                  f"最大值={E_fag.max().item():.6f}, 最小值={E_fag.min().item():.6f}")
            
            # 检查优化是否执行了（通过loss_list）
            if loss_list and len(loss_list) > 0:
                print(f"  优化执行了 {len(loss_list)} 步")
                print(f"  初始loss: {loss_list[0]:.6f}, 最终loss: {loss_list[-1]:.6f}")
            
            results['fag_adatr'] = (E_fag, X_fag)
            # 保存结果
            if output_folder and base_name:
                save_adversarial_noise(E_fag, X_fag, output_folder, base_name, 'fag_adatr', epsilon)
        except Exception as e:
            print(f"FAG-AdaTR生成失败: {e}")
            import traceback
            traceback.print_exc()
            results['fag_adatr'] = None
    
    # ATTR
    print("生成ATTR对抗噪声...")
    if not force_regenerate and output_folder and base_name:
        cached = load_adversarial_noise(output_folder, base_name, 'attr', epsilon, device)
        if cached is not None:
            results['attr'] = cached
        else:
            try:
                E_attr, _, _ = optimize_perturbation_comparison(
                    X.clone().detach(),
                    clone_tr_cores(tr_cores_init),
                    config,
                )
                X_attr = X + E_attr
                results['attr'] = (E_attr, X_attr)
                print(f"ATTR扰动范数: {torch.norm(E_attr).item():.6f}")
                # 保存结果
                if output_folder and base_name:
                    save_adversarial_noise(E_attr, X_attr, output_folder, base_name, 'attr', epsilon)
            except Exception as e:
                print(f"ATTR生成失败: {e}")
                results['attr'] = None
    else:
        try:
            E_attr, _, _ = optimize_perturbation_comparison(
                X.clone().detach(),
                clone_tr_cores(tr_cores_init),
                config,
            )
            X_attr = X + E_attr
            results['attr'] = (E_attr, X_attr)
            print(f"ATTR扰动范数: {torch.norm(E_attr).item():.6f}")
            # 保存结果
            if output_folder and base_name:
                save_adversarial_noise(E_attr, X_attr, output_folder, base_name, 'attr', epsilon)
        except Exception as e:
            print(f"ATTR生成失败: {e}")
            results['attr'] = None
    
    # 高斯噪声攻击（作为对照）
    print("生成高斯噪声攻击...")
    if not force_regenerate and output_folder and base_name:
        cached = load_adversarial_noise(output_folder, base_name, 'gaussian', epsilon, device)
        if cached is not None:
            results['gaussian'] = cached
        else:
            # 生成高斯噪声，使其范数满足epsilon约束
            # epsilon是Frobenius范数的平方，所以需要sqrt(epsilon)作为范数
            gaussian_noise = torch.randn_like(X)
            target_norm = (epsilon ** 0.5)  # sqrt(epsilon)
            current_norm = torch.norm(gaussian_noise).item()
            if current_norm > 1e-10:
                scaling_factor = target_norm / current_norm
                gaussian_noise = gaussian_noise * scaling_factor
            E_gaussian = gaussian_noise
            X_gaussian = X + E_gaussian
            results['gaussian'] = (E_gaussian, X_gaussian)
            print(f"高斯噪声扰动范数: {torch.norm(E_gaussian).item():.6f} (目标: {target_norm:.6f})")
            # 保存结果
            if output_folder and base_name:
                save_adversarial_noise(E_gaussian, X_gaussian, output_folder, base_name, 'gaussian', epsilon)
    else:
        # 生成高斯噪声，使其范数满足epsilon约束
        gaussian_noise = torch.randn_like(X)
        target_norm = (epsilon ** 0.5)  # sqrt(epsilon)
        current_norm = torch.norm(gaussian_noise).item()
        if current_norm > 1e-10:
            scaling_factor = target_norm / current_norm
            gaussian_noise = gaussian_noise * scaling_factor
        E_gaussian = gaussian_noise
        X_gaussian = X + E_gaussian
        results['gaussian'] = (E_gaussian, X_gaussian)
        print(f"高斯噪声扰动范数: {torch.norm(E_gaussian).item():.6f} (目标: {target_norm:.6f})")
        # 保存结果
        if output_folder and base_name:
            save_adversarial_noise(E_gaussian, X_gaussian, output_folder, base_name, 'gaussian', epsilon)
    
    return results


# ==================== 评估函数 ====================

def evaluate_denoising(
    original: torch.Tensor,
    noisy: torch.Tensor,
    denoised: torch.Tensor,
) -> Dict[str, float]:
    """评估去噪效果"""
    original = original.clamp(0.0, 1.0)
    noisy = noisy.clamp(0.0, 1.0)
    denoised = denoised.clamp(0.0, 1.0)
    
    # 转换为numpy数组用于calculate_psnr（如果它是tensor）
    def to_numpy(t):
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy()
        return t
    
    # PSNR
    psnr_noisy = calculate_psnr(to_numpy(original), to_numpy(noisy))
    psnr_denoised = calculate_psnr(to_numpy(original), to_numpy(denoised))
    
    # 相对误差
    rel_error_noisy = torch.norm(original - noisy).item() / (torch.norm(original).item() + 1e-10)
    rel_error_denoised = torch.norm(original - denoised).item() / (torch.norm(original).item() + 1e-10)
    
    # MSE
    mse_noisy = F.mse_loss(original, noisy).item()
    mse_denoised = F.mse_loss(original, denoised).item()
    
    return {
        'psnr_noisy': psnr_noisy,
        'psnr_denoised': psnr_denoised,
        'psnr_improvement': psnr_denoised - psnr_noisy,
        'rel_error_noisy': rel_error_noisy,
        'rel_error_denoised': rel_error_denoised,
        'rel_error_reduction': rel_error_noisy - rel_error_denoised,
        'mse_noisy': mse_noisy,
        'mse_denoised': mse_denoised,
        'mse_reduction': mse_noisy - mse_denoised,
    }


# ==================== 主实验函数 ====================

def run_diffusion_denoise_experiment(
    X: torch.Tensor,
    base_name: str,
    output_folder: str,
    config: OptimizationConfig,
    use_pretrained: bool = False,
    diffusion_steps: int = 50,
    diffusion_model_name: Optional[str] = None,
    diffusion_noise_level: float = 0.3,
    diffusion_start_step_ratio: float = 0.3,
    diffusion_guidance_scale: float = 1.0,
    force_regenerate: bool = False,
):
    """
    运行diffusion去噪实验
    
    Args:
        X: 原始张量/图像
        base_name: 基础名称（用于保存结果）
        output_folder: 输出文件夹
        config: 优化配置
        use_pretrained: 是否使用预训练模型（目前不支持，使用随机初始化的模型）
        diffusion_steps: diffusion去噪步数
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # 确保X在正确的设备上
    X = X.to(device).float()
    if X.max() > 1.0:
        X = X / 255.0
    X = X.clamp(0.0, 1.0)
    
    # 统一图像格式：确保是 [C, H, W] 格式
    # 如果通道在最后一个维度 [H, W, C]，转换为 [C, H, W]
    if X.dim() == 3:
        if X.shape[-1] == 3 or X.shape[-1] == 1:  # 通道在最后
            X = X.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        # 如果是灰度图，添加通道维度
        if X.shape[0] == 1:
            X = X.repeat(3, 1, 1)
        elif X.shape[0] > 3:
            X = X[:3]  # 只取前3个通道
    
    print(f"图像格式: {X.shape} (应该是 [C, H, W])")
    
    # 初始化TR核心（注意：TR分解期望的格式可能不同，需要检查）
    ranks = [3, 3, 3, 3]
    ranks = ranks + [ranks[0]]
    # 对于TR分解，可能需要将图像reshape为3D或4D张量
    # 这里我们先将[C, H, W]转换为[H, W, C]格式用于TR分解
    X_for_tr = X.permute(1, 2, 0) if X.dim() == 3 else X  # [C, H, W] -> [H, W, C]
    tr_cores_init = initialize_tr_cores_svd(X_for_tr, ranks, device=device)
    
    # 生成对抗噪声（使用[H, W, C]格式，因为TR分解函数期望这种格式）
    print(f"\n{'='*60}")
    print(f"为 {base_name} 生成对抗噪声")
    print(f"{'='*60}")
    # 传递output_folder和base_name以便保存/加载缓存
    adv_results = generate_adversarial_perturbations(
        X_for_tr, tr_cores_init, config,
        output_folder=output_folder,
        base_name=base_name,
        force_regenerate=force_regenerate,  # 使用传入的参数
    )
    
    # 初始化去噪器
    print("\n初始化去噪器...")
    
    # 根据参数选择去噪方法
    if use_pretrained and DIFFUSERS_AVAILABLE:
        try:
            print("使用预训练Diffusion模型...")
            print("⚠️  注意：标准DDPM模型（如google/ddpm-*）是用于生成图像的，不是专门用于去噪的。")
            print("⚠️  对于对抗噪声去噪，效果可能不理想。建议使用专门训练的去噪模型。")
            model_name = diffusion_model_name or "google/ddpm-celebahq-256"
            # 可以尝试不同的预训练模型
            # 注意：某些模型可能需要特定的图像尺寸
            diffusion_denoiser = PretrainedDiffusionDenoiser(
                model_name=model_name,
                num_inference_steps=diffusion_steps,
                noise_level=diffusion_noise_level,
                start_step_ratio=diffusion_start_step_ratio,
                guidance_scale=diffusion_guidance_scale,
            )
            print("预训练Diffusion模型加载成功！")
        except Exception as e:
            print(f"预训练Diffusion模型加载失败: {e}")
            print("回退到简单去噪网络...")
            diffusion_denoiser = SimpleDiffusionDenoiser(use_iterative=False, num_iterations=5)
    else:
        if use_pretrained:
            print("警告: diffusers库未安装，使用简单去噪网络。")
            print("安装命令: pip install diffusers transformers accelerate")
        else:
            print("使用简单去噪网络（未训练，随机初始化）...")
            print("⚠️  注意：未训练的网络效果会很差，仅作为baseline。")
        diffusion_denoiser = SimpleDiffusionDenoiser(use_iterative=False, num_iterations=5)
    
    gaussian_denoiser = SimpleGaussianDenoiser(sigma=0.1)
    
    # X现在应该是 [C, H, W] 格式，不需要再转换
    # 但我们需要处理对抗噪声结果，它们可能是 [H, W, C] 格式
    
    # 检查图像尺寸，如果太小则resize（U-Net需要至少32x32，且尺寸必须是偶数）
    min_size = 32
    if X.dim() == 3:
        C, H, W = X.shape
        if H < min_size or W < min_size:
            # 计算新的尺寸，保持宽高比，并确保是偶数（便于U-Net的pooling和upsampling对齐）
            scale = max(min_size / H, min_size / W)
            new_H = max(min_size, int(H * scale))
            new_W = max(min_size, int(W * scale))
            # 确保尺寸是偶数
            new_H = (new_H // 2) * 2
            new_W = (new_W // 2) * 2
            X = F.interpolate(
                X.unsqueeze(0),
                size=(new_H, new_W),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            print(f"原始图像尺寸过小 ({H}x{W})，已resize到 ({new_H}x{new_W})")
    
    # 对每种对抗噪声进行去噪实验
    all_results = {}
    
    for attack_name, attack_result in adv_results.items():
        if attack_result is None:
            continue
        
        E, X_adv = attack_result
        
        print(f"\n处理 {attack_name.upper()} 攻击...")
        print(f"  对抗噪声E的统计信息: 范数={torch.norm(E).item():.6f}, "
              f"均值={E.mean().item():.6f}, 标准差={E.std().item():.6f}, "
              f"最大值={E.max().item():.6f}, 最小值={E.min().item():.6f}")
        print(f"  对抗图像X_adv的统计信息: 范数={torch.norm(X_adv).item():.6f}, "
              f"均值={X_adv.mean().item():.6f}, 标准差={X_adv.std().item():.6f}")
        
        # X_adv 可能是 [H, W, C] 格式（因为TR分解函数返回这种格式）
        # 需要转换为 [C, H, W] 格式以匹配X
        if X_adv.dim() == 3:
            # 检查通道是否在最后一个维度
            if X_adv.shape[-1] == 3 or X_adv.shape[-1] == 1:
                X_adv = X_adv.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
            # 如果是灰度图，添加通道维度
            if X_adv.shape[0] == 1:
                X_adv = X_adv.repeat(3, 1, 1)
            elif X_adv.shape[0] > 3:
                X_adv = X_adv[:3]  # 只取前3个通道
        
        # 确保X_adv在[0,1]范围内
        X_adv = X_adv.clamp(0.0, 1.0)
        
        # 确保X_adv的尺寸与X完全一致（重要！）
        if X_adv.shape != X.shape:
            # 如果尺寸不匹配，resize到X的尺寸
            target_C, target_H, target_W = X.shape
            if X_adv.dim() == 3:
                X_adv = F.interpolate(
                    X_adv.unsqueeze(0),
                    size=(target_H, target_W),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
                print(f"  调整对抗图像尺寸以匹配原始图像: {X_adv.shape} (目标: {X.shape})")
        
        # Diffusion去噪
        print(f"  使用去噪网络去噪...")
        start_time = time.time()
        X_denoised_diffusion = diffusion_denoiser.denoise(X_adv.clone())
        # 确保去掉batch维度（如果有）
        if X_denoised_diffusion.dim() == 4:
            X_denoised_diffusion = X_denoised_diffusion.squeeze(0)
        diffusion_time = time.time() - start_time
        
        # 高斯去噪（baseline）
        print(f"  使用高斯去噪...")
        start_time = time.time()
        X_denoised_gaussian = gaussian_denoiser.denoise(X_adv.clone())
        # 确保去掉batch维度（如果有）
        if X_denoised_gaussian.dim() == 4:
            X_denoised_gaussian = X_denoised_gaussian.squeeze(0)
        gaussian_time = time.time() - start_time
        
        # 评估结果
        eval_diffusion = evaluate_denoising(X, X_adv, X_denoised_diffusion)
        eval_gaussian = evaluate_denoising(X, X_adv, X_denoised_gaussian)
        
        all_results[attack_name] = {
            'diffusion': eval_diffusion,
            'gaussian': eval_gaussian,
            'diffusion_time': diffusion_time,
            'gaussian_time': gaussian_time,
        }
        
        print(f"  Diffusion去噪 - PSNR: {eval_diffusion['psnr_denoised']:.2f} dB "
              f"(提升: {eval_diffusion['psnr_improvement']:.2f} dB)")
        print(f"  高斯去噪 - PSNR: {eval_gaussian['psnr_denoised']:.2f} dB "
              f"(提升: {eval_gaussian['psnr_improvement']:.2f} dB)")
        
        # 保存可视化结果
        if X.dim() == 3 and X.shape[0] == 3:
            save_comparison_images(
                X, X_adv, X_denoised_diffusion, X_denoised_gaussian,
                attack_name, base_name, output_folder
            )
    
    # 打印总结表格
    print_summary_table(all_results, base_name)
    
    # 保存结果到文件
    save_results_to_file(all_results, base_name, output_folder)
    
    return all_results


def save_comparison_images(
    original: torch.Tensor,
    noisy: torch.Tensor,
    denoised_diffusion: torch.Tensor,
    denoised_gaussian: torch.Tensor,
    attack_name: str,
    base_name: str,
    output_folder: str,
):
    """保存对比图像"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    def tensor_to_numpy(img):
        # 确保是tensor
        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)
        
        # 去掉batch维度（如果有）
        if img.dim() == 4:
            img = img.squeeze(0)
        
        # 转换为numpy并处理维度
        if img.dim() == 3:
            # [C, H, W] -> [H, W, C]
            if img.shape[0] == 3 or img.shape[0] == 1:
                img = img.permute(1, 2, 0)
            # 如果是单通道，去掉通道维度
            if img.shape[-1] == 1:
                img = img.squeeze(-1)
        elif img.dim() == 2:
            # 已经是[H, W]
            pass
        else:
            raise ValueError(f"不支持的图像维度: {img.shape}")
        
        return img.detach().cpu().numpy().clip(0, 1)
    
    axes[0, 0].imshow(tensor_to_numpy(original))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(tensor_to_numpy(noisy))
    axes[0, 1].set_title(f'{attack_name.upper()}')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(tensor_to_numpy(denoised_diffusion))
    axes[1, 0].set_title('Diffusion Denoised')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(tensor_to_numpy(denoised_gaussian))
    axes[1, 1].set_title('Gaussian Denoised')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(output_folder, f"{base_name}_{attack_name}_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存对比图像: {save_path}")


def print_summary_table(results: Dict, base_name: str):
    """打印结果总结表格"""
    print(f"\n{'='*80}")
    print(f"去噪结果总结 - {base_name}")
    print(f"{'='*80}")
    print(f"{'攻击方法':<15} {'方法':<15} {'PSNR (dB)':<15} {'PSNR提升':<15} {'相对误差':<15}")
    print(f"{'-'*80}")
    
    for attack_name, attack_results in results.items():
        for method_name, metrics in [('Diffusion', attack_results['diffusion']), 
                                     ('Gaussian', attack_results['gaussian'])]:
            print(f"{attack_name.upper():<15} {method_name:<15} "
                  f"{metrics['psnr_denoised']:<15.2f} "
                  f"{metrics['psnr_improvement']:<15.2f} "
                  f"{metrics['rel_error_denoised']:<15.6f}")
    print(f"{'='*80}\n")


def save_results_to_file(results: Dict, base_name: str, output_folder: str):
    """保存结果到CSV文件"""
    import csv
    
    csv_path = os.path.join(output_folder, f"{base_name}_denoise_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Attack', 'Method', 'PSNR_Noisy', 'PSNR_Denoised', 'PSNR_Improvement',
                        'RelError_Noisy', 'RelError_Denoised', 'RelError_Reduction',
                        'MSE_Noisy', 'MSE_Denoised', 'MSE_Reduction', 'Time'])
        
        for attack_name, attack_results in results.items():
            for method_name, metrics_key in [('Diffusion', 'diffusion'), ('Gaussian', 'gaussian')]:
                metrics = attack_results[metrics_key]
                time_key = f'{method_name.lower()}_time'
                writer.writerow([
                    attack_name.upper(),
                    method_name,
                    metrics['psnr_noisy'],
                    metrics['psnr_denoised'],
                    metrics['psnr_improvement'],
                    metrics['rel_error_noisy'],
                    metrics['rel_error_denoised'],
                    metrics['rel_error_reduction'],
                    metrics['mse_noisy'],
                    metrics['mse_denoised'],
                    metrics['mse_reduction'],
                    attack_results.get(time_key, 0),
                ])
    
    print(f"结果已保存到: {csv_path}")


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="Diffusion去噪实验：验证diffusion是否可以去除对抗噪声")
    parser.add_argument(
        "--input-folder",
        default="./AT_TD/My picture",
        help="输入图像文件夹",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="起始图像索引",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=1,
        help="结束图像索引",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=256,
        help="调整图像宽度（保持宽高比）",
    )
    parser.add_argument(
        "--output-folder",
        default="./AT_TD/diffusion_denoise_results",
        help="输出文件夹",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="对抗扰动的Frobenius范数约束 (||E||_F^2 <= epsilon)。"
             "实际扰动范数 = sqrt(epsilon)。"
             "常用值: 0.01(弱), 0.1(中等), 0.5(强), 1.0(很强)。"
             "值越大，攻击噪声越强，图像失真越明显。",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=50,
        help="Diffusion去噪步数（更多步数通常效果更好但更慢，建议50-200）",
    )
    parser.add_argument(
        "--diffusion-noise-level",
        type=float,
        default=0.3,
        help="添加的噪声水平（0.0-1.0），用于模拟diffusion过程。"
             "较小值（0.1-0.3）适合轻微噪声，较大值（0.5-0.8）适合强噪声。",
    )
    parser.add_argument(
        "--diffusion-start-ratio",
        type=float,
        default=0.3,
        help="从哪个时间步开始去噪（0.0-1.0）。"
             "0.0=从纯噪声开始（完全去噪），0.3=从30%时间步开始，1.0=几乎不去噪。"
             "对于对抗噪声，建议0.2-0.5。",
    )
    parser.add_argument(
        "--diffusion-guidance",
        type=float,
        default=1.0,
        help="引导强度（通常1.0，可以尝试1.0-3.0）。"
             "值越大，去噪越激进，但可能过度平滑。",
    )
    parser.add_argument(
        "--outer-attack-steps",
        type=int,
        default=10,
        help="攻击外循环步数",
    )
    parser.add_argument(
        "--inner-steps",
        type=int,
        default=100,
        help="内循环步数",
    )
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="使用预训练的diffusion模型（需要安装diffusers库）",
    )
    parser.add_argument(
        "--diffusion-model",
        type=str,
        default=None,
        help="预训练diffusion模型名称（如 'google/ddpm-celebahq-256'）。如果不指定，将使用默认模型",
    )
    parser.add_argument(
        "--denoiser-type",
        type=str,
        choices=["diffusion", "dncnn", "simple"],
        default="simple",
        help="去噪器类型: 'diffusion'=预训练diffusion模型, 'dncnn'=DnCNN, 'simple'=简单未训练网络",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="指定使用的GPU卡号（物理卡号）。例如 --gpu 8 表示使用第8张卡（物理卡号）。"
             "如果不指定，默认使用AT_AdaTR2的设置（通常是第8张卡）。",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="强制重新生成对抗噪声，忽略已保存的缓存。"
             "默认情况下，如果存在缓存文件，会直接加载，节省时间。",
    )
    
    args = parser.parse_args()
    
    # 设置GPU（优先级：命令行参数 > 环境变量 > AT_AdaTR2的默认设置）
    # 注意：如果已经在导入前设置了GPU（通过_gpu_arg），这里只需要验证
    if args.gpu is not None:
        if _gpu_arg is not None and _gpu_arg != args.gpu:
            print(f"⚠️  警告: 命令行参数 --gpu {args.gpu} 与导入前设置的 {_gpu_arg} 不一致")
            print(f"   由于CUDA已在导入时初始化，将使用导入前设置的GPU: {_gpu_arg}")
        elif _gpu_arg is None:
            # 如果没有在导入前设置，现在设置（但可能已经太晚了）
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            print(f"✓ 使用命令行指定的GPU: {args.gpu} (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")
            print(f"⚠️  注意: 如果CUDA已经初始化，此设置可能不会生效。建议确保--gpu参数在导入前被处理。")
        else:
            print(f"✓ 使用命令行指定的GPU: {args.gpu} (已在导入前设置)")
    elif _gpu_arg is not None:
        print(f"✓ 使用导入前设置的GPU: {_gpu_arg}")
    elif _cuda_visible_devices_before_import is not None:
        # 如果用户在导入前设置了环境变量，恢复它
        os.environ["CUDA_VISIBLE_DEVICES"] = _cuda_visible_devices_before_import
        print(f"✓ 使用环境变量指定的GPU: {_cuda_visible_devices_before_import} (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})")
    else:
        # 否则使用AT_AdaTR2设置的默认值（通常是第8张卡）
        current_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "未设置")
        print(f"✓ 使用默认GPU设置: {current_gpu} (来自AT_AdaTR2，通常是第8张卡)")
    
    # 更新全局device变量
    global device
    # 如果CUDA可用，重新检查设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ 设备已设置为: {device}")
        print(f"✓ CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        print(f"✓ 当前可见的GPU索引: {torch.cuda.current_device()}")
        print(f"✓ GPU名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        # 显示物理GPU编号（如果CUDA_VISIBLE_DEVICES设置了）
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible:
            print(f"✓ 物理GPU编号: {cuda_visible} (逻辑GPU索引: {torch.cuda.current_device()})")
    else:
        device = torch.device("cpu")
        print(f"✓ 设备已设置为: {device} (CUDA不可用)")
    
    # 创建输出文件夹
    os.makedirs(args.output_folder, exist_ok=True)
    
    # 配置
    config = OptimizationConfig(
        epsilon=args.epsilon,
        lr=0.01,
        outer_num_steps_attack=args.outer_attack_steps,
        outer_num_steps_defence=10,
        inner_num_steps=args.inner_steps,
        outer_dis_num=10,
        inner_dis_num=10,
        inner_tol=1e-10,
        attack_lr=0.01,
        defence_lr=0.01,
    )
    
    # 处理图像
    for idx in range(args.start_index, args.end_index + 1):
        image_name = f'{idx}.png'
        image_path = os.path.join(args.input_folder, image_name)
        
        if not os.path.isfile(image_path):
            print(f"图像不存在，跳过: {image_path}")
            continue
        
        print(f"\n{'#'*80}")
        print(f"处理图像: {image_name}")
        print(f"{'#'*80}")
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        orig_width, orig_height = image.size
        
        # 如果使用预训练DDPM模型，需要正方形输入（256x256）
        # 否则，按宽度resize，保持宽高比
        if args.use_pretrained or (args.denoiser_type == "diffusion"):
            # 对于DDPM模型，resize到正方形（保持宽高比，然后padding或crop）
            target_size = args.resize_width  # 默认256
            # 计算缩放比例，使最大边等于target_size
            scale = target_size / max(orig_width, orig_height)
            new_width = max(1, int(round(orig_width * scale)))
            new_height = max(1, int(round(orig_height * scale)))
            image = image.resize((new_width, new_height), Image.BILINEAR)
            
            # 如果还不是正方形，进行padding（居中padding）
            if new_width != target_size or new_height != target_size:
                # 创建256x256的黑色背景
                square_image = Image.new('RGB', (target_size, target_size), (0, 0, 0))
                # 计算居中位置
                paste_x = (target_size - new_width) // 2
                paste_y = (target_size - new_height) // 2
                square_image.paste(image, (paste_x, paste_y))
                image = square_image
                print(f"图片已resize并padding到正方形: {target_size}x{target_size} (原始: {orig_width}x{orig_height})")
        else:
            # 非DDPM模型，按宽度resize，保持宽高比
            scale = args.resize_width / orig_width
            new_height = max(1, int(round(orig_height * scale)))
            image = image.resize((args.resize_width, new_height), Image.BILINEAR)
        
        # 转换为张量
        X = torch.tensor(np.array(image) / 255.0, dtype=torch.float32)
        X = X.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
        
        base_name = os.path.basename(image_path).split('.')[0]
        
        # 运行实验
        try:
            run_diffusion_denoise_experiment(
                X,
                base_name,
                args.output_folder,
                config,
                use_pretrained=args.use_pretrained or (args.denoiser_type == "diffusion"),
                diffusion_steps=args.diffusion_steps,
                diffusion_model_name=args.diffusion_model,
                diffusion_noise_level=args.diffusion_noise_level,
                diffusion_start_step_ratio=args.diffusion_start_ratio,
                diffusion_guidance_scale=args.diffusion_guidance,
                force_regenerate=args.force_regenerate,
            )
        except Exception as e:
            print(f"处理 {image_name} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n实验完成！")


if __name__ == "__main__":
    main()
