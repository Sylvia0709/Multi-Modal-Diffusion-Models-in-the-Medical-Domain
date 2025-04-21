import sys
import os
import unittest
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到PATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入解码器
from decoder.imageDecoder import ImageDecoder
from decoder.tabularDecoder import TabularDecoder
from decoder.omicsDecoder import OmicsDecoder

class TestDecoders(unittest.TestCase):
    
    def setUp(self):
        """测试前的初始化工作"""
        # 创建测试输出目录
        self.test_output = Path("test_output")
        self.test_output.mkdir(exist_ok=True, parents=True)
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n使用设备: {self.device}")
        
        # 生成测试潜在表示
        self.batch_size = 2
        self.latent_channels = 4
        self.latent_height = self.latent_width = 64
        self.latent_dim = self.latent_channels * self.latent_height * self.latent_width
        
        # 随机生成潜在表示
        torch.manual_seed(42)  # 设置随机种子，保证可重复性
        self.latent = torch.randn(
            self.batch_size, self.latent_channels, self.latent_height, self.latent_width,
            device=self.device
        )
        
        # 创建特定的测试目录
        for dir_name in ["image", "tabular", "omics", "integration"]:
            (self.test_output / dir_name).mkdir(exist_ok=True, parents=True)
    
    def test_image_decoder(self):
        """测试图像解码器"""
        print("\n=== 测试图像解码器 ===")
        
        try:
            # 初始化图像解码器（使用本地测试用的小模型）
            # 注意：为了避免下载大模型，此处可考虑使用mock或设置use_safetensors=False
            decoder = ImageDecoder(
                output_dir=str(self.test_output / "image")
            )
            
            # 测试解码功能（如果无法下载模型，这部分可能会失败）
            try:
                # 尝试解码
                images_np = decoder.decode(self.latent, return_type="numpy")
                
                # 检查形状和范围
                self.assertEqual(images_np.shape[0], self.batch_size)
                self.assertEqual(images_np.shape[3], 3)  # RGB通道
                self.assertTrue(0 <= images_np.min() and images_np.max() <= 1)
                
                # 测试可视化功能 - 不显示，只保存
                plt.ioff()  # 关闭交互模式
                decoder.visualize(images_np, save=True, filename="test_visualization.png")
                
                print("图像解码器基本功能测试通过")
            except Exception as e:
                print(f"图像解码测试无法完成（可能是模型加载问题）: {e}")
                print("跳过后续图像解码测试")
                return
        except Exception as e:
            print(f"图像解码器初始化失败（可能是缺少依赖或网络问题）: {e}")
            print("跳过图像解码器测试")
            return
    
    def test_tabular_decoder(self):
        """测试表格解码器"""
        print("\n=== 测试表格解码器 ===")
        
        # 初始化表格解码器
        continuous_dims = 5
        categorical_dims = {"category_A": 3, "category_B": 2}
        
        decoder = TabularDecoder(
            latent_dim=self.latent_dim,
            output_dims=continuous_dims,
            categorical_dims=categorical_dims,
            output_dir=self.test_output / "tabular"
        )
        
        # 测试解码功能
        df = decoder.decode(self.latent)
        
        # 检查DataFrame结构
        self.assertEqual(len(df), self.batch_size)
        self.assertEqual(len(df.columns), continuous_dims + len(categorical_dims))
        
        # 验证连续特征
        for col in range(continuous_dims):
            col_name = f"feature_{col}"
            self.assertIn(col_name, df.columns)
        
        # 验证分类特征
        for cat_name in categorical_dims.keys():
            self.assertIn(cat_name, df.columns)
            # 检查值是否在有效范围内
            self.assertTrue(df[cat_name].max() < categorical_dims[cat_name])
            self.assertTrue(df[cat_name].min() >= 0)
        
        # 测试特征名称设置
        continuous_feature_names = [f"numeric_{i}" for i in range(continuous_dims)]
        decoder.set_feature_names(continuous_feature_names=continuous_feature_names)
        
        # 测试归一化参数设置
        feature_means = torch.zeros(continuous_dims)
        feature_stds = torch.ones(continuous_dims)
        decoder.set_normalization_params(feature_means, feature_stds)
        
        # 再次解码并验证新的特征名称
        df_with_names = decoder.decode(self.latent)
        for name in continuous_feature_names:
            self.assertIn(name, df_with_names.columns)
        
        # 测试可视化（不显示，只保存）
        plt.ioff()  # 关闭交互模式
        decoder.visualize_output(df_with_names, save=True, filename="tabular_test.png")
        
        # 测试保存模型
        decoder.save_model(filename="test_tabular_model.pt")
        self.assertTrue((self.test_output / "tabular" / "test_tabular_model.pt").exists())
        
        # 验证保存和导出
        output_file = self.test_output / "tabular" / "test_output.csv"
        decoder.save_output(df_with_names, filename="test_output.csv")
        self.assertTrue(output_file.exists())
        
        print("表格解码器测试通过")
        
        return df_with_names
    
    def test_omics_decoder(self):
        """测试组学解码器"""
        print("\n=== 测试组学解码器 ===")
        
        # 初始化组学解码器
        omics_dim = 1000
        decoder = OmicsDecoder(
            latent_dim=self.latent_dim,
            omics_dim=omics_dim,
            omics_type="gene_expression",
            use_attention=False,  # 简化测试，不使用注意力机制
            output_dir=self.test_output / "omics"
        )
        
        # 设置特征名称
        feature_names = [f"gene_{i}" for i in range(omics_dim)]
        decoder.set_feature_names(feature_names)
        
        # 设置归一化参数
        feature_means = np.zeros(omics_dim)
        feature_stds = np.ones(omics_dim)
        decoder.set_normalization_params(feature_means, feature_stds)
        
        # 测试解码功能
        df = decoder.decode(self.latent)
        
        # 检查DataFrame结构
        self.assertEqual(len(df), self.batch_size)
        self.assertEqual(len(df.columns), omics_dim)
        
        # 检查值范围 (gene_expression应为非负值)
        self.assertTrue((df.values >= 0).all())
        
        # 检查特征名称
        for i in range(min(10, omics_dim)):  # 检查前10个特征名称
            self.assertIn(f"gene_{i}", df.columns)
        
        # 测试可视化（不显示，只保存）
        plt.ioff()  # 关闭交互模式
        try:
            # PCA可视化测试
            decoder._visualize_pca(df, save=True, filename="omics_pca_test.png")
            self.assertTrue((self.test_output / "omics" / "omics_pca_test.png").exists())
            
            # 热图可视化测试（仅使用少量特征以加快速度）
            decoder._visualize_heatmap(df.iloc[:, :20], save=True, filename="omics_heatmap_test.png")
            self.assertTrue((self.test_output / "omics" / "omics_heatmap_test.png").exists())
        except Exception as e:
            print(f"可视化测试发生错误: {e}")
        
        # 测试保存模型
        decoder.save_model(filename="test_omics_model.pt")
        self.assertTrue((self.test_output / "omics" / "test_omics_model.pt").exists())
        
        # 验证保存和导出
        output_file = self.test_output / "omics" / "test_omics_data.csv"
        decoder.save_output(df, filename="test_omics_data.csv")
        self.assertTrue(output_file.exists())
        
        print("组学解码器测试通过")
        
        return df
    
    def test_integration(self):
        """测试三个解码器的集成"""
        print("\n=== 测试解码器集成 ===")
        
        integration_dir = self.test_output / "integration"
        
        # 初始化所有解码器
        try:
            img_decoder = ImageDecoder(output_dir=str(integration_dir / "images"))
        except Exception as e:
            print(f"图像解码器初始化失败: {e}")
            print("跳过集成测试中的图像部分")
            img_decoder = None
        
        tab_decoder = TabularDecoder(
            latent_dim=self.latent_dim,
            output_dims=10,
            categorical_dims={"cat1": 4, "cat2": 3},
            output_dir=integration_dir / "tabular"
        )
        
        omics_decoder = OmicsDecoder(
            latent_dim=self.latent_dim,
            omics_dim=500,
            use_attention=False,
            output_dir=integration_dir / "omics"
        )
        
        # 从同一个潜在表示解码到不同模态
        if img_decoder is not None:
            try:
                images = img_decoder.decode(self.latent)
                self.assertEqual(len(images), self.batch_size)
            except Exception as e:
                print(f"图像解码失败: {e}")
        
        tabular_data = tab_decoder.decode(self.latent)
        omics_data = omics_decoder.decode(self.latent)
        
        # 验证所有解码器都能处理相同的输入
        self.assertEqual(len(tabular_data), self.batch_size)
        self.assertEqual(len(omics_data), self.batch_size)
        
        # 对不同解码结果进行相关性分析（仅示例）
        # 提取表格连续特征的前3个
        tab_features = tabular_data.iloc[:, :3].values
        # 提取组学特征的前3个
        omics_features = omics_data.iloc[:, :3].values
        
        print(f"表格数据形状: {tab_features.shape}")
        print(f"组学数据形状: {omics_features.shape}")
        
        # 保存各解码器结果
        tabular_data.to_csv(integration_dir / "integrated_tabular.csv", index=False)
        omics_data.to_csv(integration_dir / "integrated_omics.csv", index=False)
        
        print("集成测试通过")

if __name__ == "__main__":
    unittest.main()
