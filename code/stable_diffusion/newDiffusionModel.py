import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from diffusers import StableDiffusionPipeline, DDIMScheduler, LMSDiscreteScheduler
from PIL import Image
import os


class MultiModalDiffusionModel:
    """
    多模态扩散模型 - 负责从跨模态注意力输出生成潜在空间表示(latent)
    此模块作为多模态系统的Diffusion部分，连接跨模态注意力模块和各种解码器
    """
    
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device=None, output_dir="output"):
        """
        初始化多模态扩散模型
        
        Args:
            model_id: Stable Diffusion模型ID
            device: 计算设备，如果为None则自动检测
            output_dir: 输出目录
        """
        # 设置随机种子以确保可复现性
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        print(f"使用设备: {self.device}, 数据类型: {self.dtype}")
        
        # 创建输出目录
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 加载Stable Diffusion模型
        print(f"正在加载Stable Diffusion模型: {model_id}...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            use_safetensors=True
        )
        self.pipe = self.pipe.to(self.device)
        print("Stable Diffusion模型加载完成")
        
        # 初始化调度器
        self.scheduler = None
        self.setup_scheduler("ddim", 50)
        
        # 设置常量
        self.latent_channels = 4
        self.latent_height = self.latent_width = 64
        self.unet_hidden_size = 768
        self.cross_attn_dim = 256  # 默认跨模态维度，可在process_cross_attention_output中覆盖
        
    def setup_scheduler(self, scheduler_type="ddim", num_inference_steps=50):
        """
        设置和配置扩散调度器
        
        Args:
            scheduler_type: 调度器类型，支持"ddim"或"lms"
            num_inference_steps: 推理步数
        """
        if scheduler_type.lower() == "ddim":
            self.scheduler = DDIMScheduler.from_pretrained(
                "runwayml/stable-diffusion-v1-5", 
                subfolder="scheduler"
            )
        elif scheduler_type.lower() == "lms":
            self.scheduler = LMSDiscreteScheduler.from_pretrained(
                "runwayml/stable-diffusion-v1-5", 
                subfolder="scheduler"
            )
        else:
            raise ValueError(f"不支持的调度器类型: {scheduler_type}")
        
        # 明确设置时间步，并指定设备
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        self.num_inference_steps = num_inference_steps
        
        # 验证时间步是否正确设置
        if self.scheduler.timesteps is None or len(self.scheduler.timesteps) == 0:
            raise ValueError("调度器时间步未正确设置！")
        elif self.scheduler.num_inference_steps is None:
            raise ValueError("调度器的num_inference_steps仍为None！")
            
        print(f"已设置 {scheduler_type.upper()} 调度器，步数: {num_inference_steps}")
        print(f"调度器时间步数: {len(self.scheduler.timesteps)}")
        
    def process_cross_attention_output(self, cross_attention_output=None, batch_size=1):
        """
        处理跨模态注意力输出，将其转换为UNet可用的条件嵌入
        
        Args:
            cross_attention_output: 跨模态注意力输出，形状为[batch, N, dim]
            batch_size: 如果cross_attention_output为None，则使用此批大小生成随机数据
            
        Returns:
            encoder_hidden_states: UNet条件嵌入，形状为[batch, 77, 768]
        """
        # 如果未提供跨模态注意力输出，则使用随机数据模拟
        if cross_attention_output is None:
            print("警告: 使用随机数据模拟跨模态注意力输出")
            N = 1  # 假设输出1个token
            d_cross = self.cross_attn_dim
            cross_attention_output = torch.randn(
                batch_size, N, d_cross, 
                dtype=self.dtype, device=self.device
            )
        
        # 获取跨模态维度
        d_cross = cross_attention_output.shape[-1]
        
        # 创建投影层，将跨模态维度映射到UNet条件维度
        proj_layer = nn.Linear(d_cross, self.unet_hidden_size).to(self.device)
        if self.dtype == torch.float16:
            proj_layer = proj_layer.half()
        
        # 投影到UNet维度
        projected = proj_layer(cross_attention_output)  # [batch, N, 768]
        
        # 扩展到目标序列长度 (SD需要77个token)
        target_seq_len = 77
        if projected.shape[1] < target_seq_len:
            token = projected[:, 0:1, :]  # 取第一个token
            encoder_hidden_states = token.repeat(1, target_seq_len, 1)
        else:
            encoder_hidden_states = projected[:, :target_seq_len, :]
        
        # 确保数据类型正确
        if self.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.half()
            
        print(f"条件嵌入形状: {encoder_hidden_states.shape}")
        return encoder_hidden_states
        
    def generate_latent(self, cross_attention_output=None, batch_size=1, return_intermediates=False):
        """
        从跨模态注意力输出生成潜在表示
        
        Args:
            cross_attention_output: 跨模态注意力输出
            batch_size: 批大小
            return_intermediates: 是否返回中间步骤的潜在表示
            
        Returns:
            latent: 生成的潜在表示
            intermediate_latents: 如果return_intermediates=True，返回中间步骤的潜在表示
        """
        # 处理跨模态注意力输出
        encoder_hidden_states = self.process_cross_attention_output(
            cross_attention_output, batch_size
        )
        
        # 初始化潜在空间噪声
        latent = torch.randn(
            batch_size, self.latent_channels, self.latent_height, self.latent_width,
            dtype=self.dtype, device=self.device
        )
        
        # 验证调度器设置
        if self.scheduler.timesteps is None or len(self.scheduler.timesteps) == 0:
            raise ValueError("调度器时间步未正确设置！请先调用setup_scheduler方法")
        
        # 扩散采样过程
        self.pipe.unet.eval()
        intermediate_latents = []
        
        with torch.no_grad():
            for i, t in enumerate(self.scheduler.timesteps):
                # 打印进度
                if i % 10 == 0 or i == len(self.scheduler.timesteps) - 1:
                    print(f"采样步骤 {i+1}/{len(self.scheduler.timesteps)}")
                
                # 调用UNet进行噪声预测
                noise_pred = self.pipe.unet(
                    latent, t, encoder_hidden_states=encoder_hidden_states
                ).sample
                
                # 保存中间结果
                if return_intermediates and (i % 10 == 0 or i == len(self.scheduler.timesteps) - 1):
                    intermediate_latents.append(latent.detach().clone())
                
                # 扩散步骤
                latent = self.scheduler.step(noise_pred, t, latent)["prev_sample"]
        
        print(f"潜在表示生成完成，形状: {latent.shape}")
        
        if return_intermediates:
            return latent, intermediate_latents
        else:
            return latent
            
    def decode_to_image(self, latent):
        """
        将潜在表示解码为图像（示例解码器）
        
        Args:
            latent: 潜在表示
            
        Returns:
            images: 解码后的图像，形状为[B, H, W, 3]
        """
        with torch.no_grad():
            # 使用VAE解码
            images = self.pipe.vae.decode(latent / self.pipe.vae.config.scaling_factor).sample
        
        # 后处理图像
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).float().numpy()
        
        return images
        
    def visualize_latent(self, latent, save=True, prefix="generated"):
        """
        可视化潜在表示（解码为图像并显示）
        
        Args:
            latent: 潜在表示
            save: 是否保存图像
            prefix: 保存文件名前缀
        """
        # 解码为图像
        images = self.decode_to_image(latent)
        
        # 显示图像
        plt.figure(figsize=(10, 10))
        for i, img in enumerate(images):
            plt.subplot(1, len(images), i + 1)
            plt.imshow(img)
            plt.axis("off")
        
        plt.tight_layout()
        
        if save:
            # 保存可视化结果
            plt.savefig(self.output_dir / f"{prefix}_visualization.png")
            
            # 保存单独的图像文件
            for i, image in enumerate(images):
                img = Image.fromarray((image * 255).astype(np.uint8))
                img_path = self.output_dir / f"{prefix}_{i}.png"
                img.save(img_path)
                print(f"图像已保存至 {img_path}")
        
        plt.show()
        return images
        
    def visualize_diffusion_process(self, intermediate_latents, save=True):
        """
        可视化扩散过程
        
        Args:
            intermediate_latents: 中间步骤的潜在表示列表
            save: 是否保存可视化结果
        """
        if not intermediate_latents:
            print("没有中间步骤的潜在表示，无法可视化扩散过程")
            return
            
        plt.figure(figsize=(20, 4))
        for i, latent in enumerate(intermediate_latents):
            # 解码为图像
            img = self.decode_to_image(latent)[0]  # 取第一张图片
            
            plt.subplot(1, len(intermediate_latents), i + 1)
            plt.imshow(img)
            plt.axis("off")
            plt.title(f"步骤 {i*10}")
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "diffusion_process.png")
            print(f"扩散过程可视化已保存至 {self.output_dir / 'diffusion_process.png'}")
        
        plt.show()
        
    def analyze_latent(self, latent, save=True):
        """
        分析潜在表示的统计特性
        
        Args:
            latent: 潜在表示
            save: 是否保存分析结果
        """
        # 将张量转换为NumPy数组
        latent_np = latent.detach().cpu().numpy()
        
        # 计算统计信息
        mean = np.mean(latent_np)
        std = np.std(latent_np)
        min_val = np.min(latent_np)
        max_val = np.max(latent_np)
        
        print("\n潜在表示统计分析:")
        print(f"- 均值: {mean:.4f}")
        print(f"- 标准差: {std:.4f}")
        print(f"- 最小值: {min_val:.4f}")
        print(f"- 最大值: {max_val:.4f}")
        
        # 绘制分布直方图
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(latent_np.flatten(), bins=50, alpha=0.75)
        plt.title("潜在表示值分布")
        plt.xlabel("值")
        plt.ylabel("频率")
        plt.grid(alpha=0.3)
        
        # 绘制通道均值分布
        plt.subplot(1, 2, 2)
        channel_means = [np.mean(latent_np[:, c]) for c in range(latent_np.shape[1])]
        plt.bar(range(len(channel_means)), channel_means)
        plt.title("各通道均值")
        plt.xlabel("通道")
        plt.ylabel("均值")
        plt.xticks(range(len(channel_means)))
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / "latent_analysis.png")
            print(f"潜在表示分析已保存至 {self.output_dir / 'latent_analysis.png'}")
        
        plt.show()
        return {
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val,
            "channel_means": channel_means
        }


# 示例用法
if __name__ == "__main__":
    # 创建多模态扩散模型
    model = MultiModalDiffusionModel()
    
    # 生成潜在表示，包括中间步骤
    latent, intermediates = model.generate_latent(return_intermediates=True)
    
    # 可视化最终的潜在表示
    model.visualize_latent(latent)
    
    # 可视化扩散过程
    model.visualize_diffusion_process(intermediates)
    
    # 分析潜在表示
    model.analyze_latent(latent)
    
    print("\n多模态扩散模型演示完成！")
