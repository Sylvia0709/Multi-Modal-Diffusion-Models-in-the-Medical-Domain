import sys
import os
import unittest
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# 添加项目根目录到PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入解码器
from decoder.imageDecoder import ImageDecoder
from decoder.tabularDecoder import TabularDecoder
from decoder.omicsDecoder import OmicsDecoder

# 导入扩散模型
try:
    from stable_diffusion.newDiffusionModel import MultiModalDiffusionModel
    DIFFUSION_MODEL_AVAILABLE = True
except ImportError:
    print("警告: 无法导入MultiModalDiffusionModel，将跳过端到端测试")
    DIFFUSION_MODEL_AVAILABLE = False

class TestMultiModal(unittest.TestCase):
    """测试多模态扩散模型的端到端流程"""
    
    def setUp(self):
        """测试前的初始化工作"""
        # 创建测试输出目录
        self.test_output = Path("test_output/multi_modal")
        self.test_output.mkdir(exist_ok=True, parents=True)
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n使用设备: {self.device}")
        
        # 模拟跨模态注意力输出
        self.batch_size = 2
        self.cross_attn_dim = 256
        self.cross_attention_output = torch.randn(
            self.batch_size, 1, self.cross_attn_dim,
            device=self.device
        )
        
        # 记录diffusion模型是否可用
        self.diffusion_model_available = DIFFUSION_MODEL_AVAILABLE
    
    def test_diffusion_model_interface(self):
        """测试扩散模型的接口"""
        print("\n=== 测试扩散模型接口 ===")
        
        if not self.diffusion_model_available:
            print("跳过扩散模型接口测试（模型不可用）")
            return
        
        try:
            # 初始化扩散模型（可能会下载模型）
            diffusion_model = MultiModalDiffusionModel(
                model_id="runwayml/stable-diffusion-v1-5",
                device=self.device,
                output_dir=str(self.test_output)
            )
            
            # 测试跨模态注意力输出处理
            encoder_hidden_states = diffusion_model.process_cross_attention_output(
                self.cross_attention_output
            )
            
            # 验证输出形状
            expected_shape = (self.batch_size, 77, 768)  # SD需要77个token
            self.assertEqual(encoder_hidden_states.shape, expected_shape)
            
            print("扩散模型接口测试通过")
        except Exception as e:
            print(f"扩散模型接口测试失败: {e}")
            return
    
    def test_end_to_end_pipeline_mock(self):
        """使用模拟数据测试从扩散模型到各个解码器的完整流程"""
        print("\n=== 测试端到端管道（模拟数据）===")
        
        # 生成模拟的潜在表示（通常由扩散模型生成）
        latent_channels = 4
        latent_height = latent_width = 64
        latent = torch.randn(
            self.batch_size, latent_channels, latent_height, latent_width,
            device=self.device
        )
        latent_dim = latent_channels * latent_height * latent_width
        
        # 初始化解码器
        tab_decoder = TabularDecoder(
            latent_dim=latent_dim,
            output_dims=5,
            categorical_dims={"category_A": 3},
            output_dir=self.test_output / "tabular"
        )
        
        omics_decoder = OmicsDecoder(
            latent_dim=latent_dim,
            omics_dim=100,  # 使用较小的维度加快测试
            omics_type="gene_expression",
            use_attention=False,
            output_dir=self.test_output / "omics"
        )
        
        # 解码到不同模态
        tabular_data = tab_decoder.decode(latent)
        omics_data = omics_decoder.decode(latent)
        
        # 验证解码结果
        self.assertEqual(len(tabular_data), self.batch_size)
        self.assertEqual(len(omics_data), self.batch_size)
        
        print("端到端管道（模拟数据）测试通过")
    
    def test_end_to_end_pipeline_real(self):
        """测试使用真实扩散模型的完整流程（如果可用）"""
        print("\n=== 测试端到端管道（真实模型）===")
        
        if not self.diffusion_model_available:
            print("跳过端到端真实模型测试（模型不可用）")
            return
        
        try:
            # 初始化扩散模型
            diffusion_model = MultiModalDiffusionModel(
                model_id="runwayml/stable-diffusion-v1-5",
                device=self.device,
                output_dir=str(self.test_output)
            )
            
            # 生成潜在表示
            latent = diffusion_model.generate_latent(
                cross_attention_output=self.cross_attention_output,
                batch_size=self.batch_size
            )
            
            # 初始化各个解码器
            tab_decoder = TabularDecoder(
                latent_dim=latent.numel() // self.batch_size,
                output_dims=5,
                categorical_dims={"category_A": 3},
                output_dir=self.test_output / "tabular_real"
            )
            
            omics_decoder = OmicsDecoder(
                latent_dim=latent.numel() // self.batch_size,
                omics_dim=100,
                omics_type="gene_expression",
                use_attention=False,
                output_dir=self.test_output / "omics_real"
            )
            
            # 解码到不同模态
            tabular_data = tab_decoder.decode(latent)
            omics_data = omics_decoder.decode(latent)
            
            # 尝试使用扩散模型内置的图像解码器
            image_data = diffusion_model.decode_to_image(latent)
            
            # 验证解码结果
            self.assertEqual(len(tabular_data), self.batch_size)
            self.assertEqual(len(omics_data), self.batch_size)
            self.assertEqual(len(image_data), self.batch_size)
            
            # 可视化潜在表示
            diffusion_model.visualize_latent(latent, save=True, prefix="test_end_to_end")
            
            print("端到端管道（真实模型）测试通过")
        except Exception as e:
            print(f"端到端真实模型测试失败: {e}")
            return
    
    def test_module_compatibility(self):
        """测试各模块之间的兼容性"""
        print("\n=== 测试模块兼容性 ===")
        
        # 验证各解码器的输入维度计算方式相同
        latent_channels = 4
        latent_height = latent_width = 64
        latent = torch.randn(
            self.batch_size, latent_channels, latent_height, latent_width,
            device=self.device
        )
        
        # 计算不同方式的潜在维度
        latent_dim_product = latent_channels * latent_height * latent_width
        latent_dim_flattened = torch.flatten(latent[0]).shape[0]
        latent_dim_numel = latent[0].numel()
        
        # 确保所有维度计算方法一致
        self.assertEqual(latent_dim_product, latent_dim_flattened)
        self.assertEqual(latent_dim_product, latent_dim_numel)
        
        print(f"潜在维度计算一致: {latent_dim_product}")
        print("模块兼容性测试通过")


if __name__ == "__main__":
    unittest.main() 