import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import os

class ImageDecoder:
    """
    图像解码器 - 将潜在表示(latent)解码为图像
    用于多模态扩散模型框架的图像模态解码部分
    """
    
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device=None, output_dir="output/images"):
        """
        初始化图像解码器
        
        Args:
            model_id: Stable Diffusion模型ID
            device: 计算设备，如果为None则自动检测
            output_dir: 输出目录
        """
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        print(f"图像解码器使用设备: {self.device}, 数据类型: {self.dtype}")
        
        # 创建输出目录
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 加载VAE解码器（从Stable Diffusion模型中提取）
        print(f"正在加载VAE解码器 (从{model_id})...")
        # 仅加载VAE部分以节省内存
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            use_safetensors=True,
            components=["vae"]
        )
        self.vae = pipe.vae.to(self.device)
        print("VAE解码器加载完成")
    
    def decode(self, latent, normalize=True, return_type="numpy"):
        """
        将潜在表示解码为图像
        
        Args:
            latent: 潜在表示，形状为[B, 4, H, W]
            normalize: 是否将输出归一化到[0,1]范围
            return_type: 返回类型，可以是"numpy"、"tensor"或"pil"
            
        Returns:
            images: 解码后的图像
        """
        if not isinstance(latent, torch.Tensor):
            latent = torch.tensor(latent, dtype=self.dtype, device=self.device)
        
        # 确保潜在表示具有正确的形状和类型
        if latent.dim() == 3:
            # 添加批次维度
            latent = latent.unsqueeze(0)
        
        # 确保数据类型匹配
        latent = latent.to(device=self.device, dtype=self.dtype)
        
        with torch.no_grad():
            # 使用VAE解码
            images = self.vae.decode(latent / self.vae.config.scaling_factor).sample
        
        # 后处理图像
        if normalize:
            images = (images / 2 + 0.5).clamp(0, 1)
        
        if return_type == "tensor":
            return images
        
        # 转换为numpy数组，形状为[B, H, W, C]
        images_np = images.cpu().permute(0, 2, 3, 1).float().numpy()
        
        if return_type == "numpy":
            return images_np
        elif return_type == "pil":
            # 转换为PIL图像列表
            pil_images = []
            for img_array in images_np:
                img_array = (img_array * 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_array))
            return pil_images
        else:
            raise ValueError(f"不支持的返回类型: {return_type}")
    
    def save_images(self, images, prefix="generated"):
        """
        保存图像到文件
        
        Args:
            images: 图像数组或PIL图像列表
            prefix: 文件名前缀
        """
        if isinstance(images, torch.Tensor):
            images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        
        if isinstance(images, np.ndarray):
            for i, img_array in enumerate(images):
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                img = Image.fromarray(img_array)
                img_path = self.output_dir / f"{prefix}_{i}.png"
                img.save(img_path)
                print(f"图像已保存至 {img_path}")
        elif isinstance(images, list) and isinstance(images[0], Image.Image):
            for i, img in enumerate(images):
                img_path = self.output_dir / f"{prefix}_{i}.png"
                img.save(img_path)
                print(f"图像已保存至 {img_path}")
        else:
            raise ValueError("不支持的图像格式")
    
    def visualize(self, images, title="解码图像", save=True, filename="visualization.png"):
        """
        可视化图像
        
        Args:
            images: 图像数组或PIL图像列表
            title: 图像标题
            save: 是否保存可视化结果
            filename: 保存的文件名
        """
        if isinstance(images, torch.Tensor):
            images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        
        # 转换PIL图像为numpy数组
        if isinstance(images, list) and isinstance(images[0], Image.Image):
            images = [np.array(img) / 255.0 for img in images]
        
        # 创建图像网格
        n_images = len(images)
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols
        
        plt.figure(figsize=(cols * 4, rows * 4))
        for i, img in enumerate(images):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(img)
            plt.axis("off")
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / filename
            plt.savefig(save_path)
            print(f"可视化结果已保存至 {save_path}")
        
        plt.show()
    
    def enhance_image(self, image, method="super_resolution"):
        """
        增强图像质量
        
        Args:
            image: 输入图像
            method: 增强方法，可以是"super_resolution"或"color_enhancement"
            
        Returns:
            enhanced_image: 增强后的图像
        """
        # 这里可以实现不同的图像增强方法
        # 作为示例，我们暂时只实现简单的返回
        print(f"图像增强方法 '{method}' 尚未实现")
        return image


# 示例用法
if __name__ == "__main__":
    # 创建图像解码器
    decoder = ImageDecoder()
    
    # 生成随机潜在表示（通常从扩散模型获得）
    batch_size = 1
    latent_channels = 4
    latent_height = latent_width = 64
    random_latent = torch.randn(
        batch_size, latent_channels, latent_height, latent_width,
        dtype=decoder.dtype, device=decoder.device
    )
    
    # 解码为图像
    decoded_images = decoder.decode(random_latent, return_type="pil")
    
    # 可视化结果
    decoder.visualize(decoded_images, title="随机潜在表示解码结果")
    
    # 保存图像
    decoder.save_images(decoded_images, prefix="random")
    
    print("图像解码器演示完成！")
