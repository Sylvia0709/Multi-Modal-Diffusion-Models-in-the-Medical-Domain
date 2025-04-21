import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import pickle


class TabularDecoder(nn.Module):
    """
    表格数据解码器 - 将潜在表示(latent)解码为表格数据
    用于多模态扩散模型框架的表格数据模态解码部分
    """
    
    def __init__(self, latent_dim=16384, # 4*64*64
                 output_dims=None,
                 categorical_dims=None,
                 hidden_dims=[2048, 1024, 512],
                 dropout_rate=0.2,
                 activation=nn.LeakyReLU(0.2),
                 device=None,
                 output_dir="output/tabular"):
        """
        初始化表格数据解码器
        
        Args:
            latent_dim: 潜在表示的维度（对于SD通常是4*64*64=16384）
            output_dims: 连续输出特征的维度（数值特征的数量）
            categorical_dims: 分类特征的维度字典 {特征名: 类别数}
            hidden_dims: 隐藏层维度的列表
            dropout_rate: Dropout比率
            activation: 激活函数
            device: 计算设备，如果为None则自动检测
            output_dir: 输出目录
        """
        super(TabularDecoder, self).__init__()
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # 创建输出目录
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 特征配置
        self.latent_dim = latent_dim
        self.output_dims = output_dims if output_dims is not None else 10  # 默认10个连续特征
        self.categorical_dims = categorical_dims if categorical_dims is not None else {}
        
        # 计算总输出维度（连续特征 + 所有分类特征的类别总数）
        self.total_continuous_dims = self.output_dims
        self.total_categorical_dims = sum(self.categorical_dims.values())
        
        # 特征名称记录
        self.continuous_feature_names = None
        self.categorical_feature_names = list(self.categorical_dims.keys()) if self.categorical_dims else []
        
        # 记录特征归一化参数
        self.feature_means = None
        self.feature_stds = None
        
        # 构建网络
        layers = []
        
        # 展平层
        self.flatten = nn.Flatten()
        
        # 主干网络
        in_dim = latent_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation)
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # 连续特征输出头
        if self.total_continuous_dims > 0:
            self.continuous_head = nn.Linear(hidden_dims[-1], self.total_continuous_dims)
        
        # 分类特征输出头（每个分类特征一个头）
        if self.categorical_dims:
            self.categorical_heads = nn.ModuleDict()
            for feat_name, num_categories in self.categorical_dims.items():
                self.categorical_heads[feat_name] = nn.Linear(hidden_dims[-1], num_categories)
        
        # 移动模型到指定设备
        self.to(self.device)
        print(f"表格数据解码器已初始化，设备: {self.device}")
        print(f"连续特征数量: {self.total_continuous_dims}")
        print(f"分类特征数量: {len(self.categorical_dims)}")
        print(f"分类特征总类别数: {self.total_categorical_dims}")
    
    def forward(self, latent):
        """
        前向传播
        
        Args:
            latent: 潜在表示，形状为[B, C, H, W]
            
        Returns:
            continuous_output: 连续特征输出
            categorical_outputs: 分类特征输出字典 {特征名: logits}
        """
        # 确保输入是tensor
        if not isinstance(latent, torch.Tensor):
            latent = torch.tensor(latent, device=self.device)
        
        # 确保形状正确
        if latent.dim() == 3:
            latent = latent.unsqueeze(0)  # 添加批次维度
        
        # 展平潜在表示
        x = self.flatten(latent)
        
        # 通过主干网络
        features = self.backbone(x)
        
        # 输出连续特征
        continuous_output = None
        if hasattr(self, 'continuous_head'):
            continuous_output = self.continuous_head(features)
        
        # 输出分类特征
        categorical_outputs = {}
        if hasattr(self, 'categorical_heads'):
            for feat_name, head in self.categorical_heads.items():
                categorical_outputs[feat_name] = head(features)
        
        return continuous_output, categorical_outputs
    
    def decode(self, latent, denormalize=True):
        """
        将潜在表示解码为表格数据
        
        Args:
            latent: 潜在表示
            denormalize: 是否反归一化连续特征
            
        Returns:
            df: pandas DataFrame包含解码后的表格数据
        """
        with torch.no_grad():
            continuous_output, categorical_outputs = self.forward(latent)
        
        # 处理连续输出
        if continuous_output is not None:
            continuous_np = continuous_output.cpu().numpy()
            
            # 反归一化（如果有归一化参数且需要反归一化）
            if denormalize and self.feature_means is not None and self.feature_stds is not None:
                continuous_np = continuous_np * self.feature_stds + self.feature_means
        else:
            continuous_np = np.array([])
        
        # 处理分类输出（转换为最可能的类别）
        categorical_np = {}
        if categorical_outputs:
            for feat_name, logits in categorical_outputs.items():
                probs = torch.softmax(logits, dim=1)
                categories = torch.argmax(probs, dim=1).cpu().numpy()
                categorical_np[feat_name] = categories
        
        # 创建DataFrame
        batch_size = latent.shape[0] if isinstance(latent, torch.Tensor) else 1
        
        data = {}
        
        # 添加连续特征
        if continuous_output is not None and continuous_np.size > 0:
            feature_names = self.continuous_feature_names or [f"feature_{i}" for i in range(continuous_np.shape[1])]
            for i, name in enumerate(feature_names):
                if i < continuous_np.shape[1]:
                    data[name] = continuous_np[:, i]
        
        # 添加分类特征
        for feat_name, categories in categorical_np.items():
            data[feat_name] = categories
        
        return pd.DataFrame(data)
    
    def set_feature_names(self, continuous_feature_names=None, categorical_feature_names=None):
        """
        设置特征名称
        
        Args:
            continuous_feature_names: 连续特征名称列表
            categorical_feature_names: 分类特征名称列表（必须与初始化时的categorical_dims的键匹配）
        """
        if continuous_feature_names is not None:
            if len(continuous_feature_names) != self.total_continuous_dims:
                print(f"警告: 连续特征名称数量({len(continuous_feature_names)})与特征维度({self.total_continuous_dims})不匹配")
            self.continuous_feature_names = continuous_feature_names
        
        if categorical_feature_names is not None:
            if set(categorical_feature_names) != set(self.categorical_dims.keys()):
                print("警告: 提供的分类特征名称与初始化时的categorical_dims不匹配")
            else:
                self.categorical_feature_names = categorical_feature_names
    
    def set_normalization_params(self, feature_means, feature_stds):
        """
        设置特征归一化参数，用于反归一化
        
        Args:
            feature_means: 特征均值
            feature_stds: 特征标准差
        """
        self.feature_means = feature_means
        self.feature_stds = feature_stds
        print("已设置特征归一化参数，解码时将自动反归一化")
    
    def save_model(self, filename="tabular_decoder.pkl"):
        """
        保存模型和配置
        
        Args:
            filename: 保存的文件名
        """
        save_path = self.output_dir / filename
        
        model_state = {
            'model_state_dict': self.state_dict(),
            'latent_dim': self.latent_dim,
            'output_dims': self.output_dims,
            'categorical_dims': self.categorical_dims,
            'continuous_feature_names': self.continuous_feature_names,
            'categorical_feature_names': self.categorical_feature_names,
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds
        }
        
        torch.save(model_state, save_path)
        print(f"模型已保存至 {save_path}")
    
    @classmethod
    def load_model(cls, filename, device=None, output_dir="output/tabular"):
        """
        加载模型
        
        Args:
            filename: 模型文件名
            device: 计算设备
            output_dir: 输出目录
            
        Returns:
            model: 加载的模型
        """
        model_path = Path(output_dir) / filename
        
        model_state = torch.load(model_path, map_location=device)
        
        model = cls(
            latent_dim=model_state['latent_dim'],
            output_dims=model_state['output_dims'],
            categorical_dims=model_state['categorical_dims'],
            device=device,
            output_dir=output_dir
        )
        
        model.load_state_dict(model_state['model_state_dict'])
        model.continuous_feature_names = model_state['continuous_feature_names']
        model.categorical_feature_names = model_state['categorical_feature_names']
        model.feature_means = model_state['feature_means']
        model.feature_stds = model_state['feature_stds']
        
        print(f"模型已从 {model_path} 加载")
        return model
    
    def visualize_output(self, df, n_samples=None, save=True, filename="tabular_visualization.png"):
        """
        可视化解码后的表格数据
        
        Args:
            df: 解码后的DataFrame
            n_samples: 要可视化的样本数量
            save: 是否保存可视化结果
            filename: 保存的文件名
        """
        if n_samples is not None and n_samples < len(df):
            df = df.iloc[:n_samples]
        
        # 分离连续和分类特征
        categorical_cols = [col for col in df.columns if col in self.categorical_feature_names]
        continuous_cols = [col for col in df.columns if col not in categorical_cols]
        
        n_plots = len(continuous_cols) + len(categorical_cols)
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols
        
        plt.figure(figsize=(cols * 5, rows * 4))
        
        # 绘制连续特征分布
        for i, col in enumerate(continuous_cols):
            plt.subplot(rows, cols, i + 1)
            sns.histplot(df[col], kde=True)
            plt.title(f"连续特征: {col}")
            plt.tight_layout()
        
        # 绘制分类特征分布
        for i, col in enumerate(categorical_cols):
            plt.subplot(rows, cols, i + len(continuous_cols) + 1)
            counts = df[col].value_counts().sort_index()
            sns.barplot(x=counts.index, y=counts.values)
            plt.title(f"分类特征: {col}")
            plt.xticks(rotation=45)
            plt.tight_layout()
        
        if save:
            save_path = self.output_dir / filename
            plt.savefig(save_path)
            print(f"可视化结果已保存至 {save_path}")
        
        plt.show()
    
    def save_output(self, df, filename="decoded_tabular_data.csv"):
        """
        保存解码后的表格数据
        
        Args:
            df: 解码后的DataFrame
            filename: 保存的文件名
        """
        save_path = self.output_dir / filename
        df.to_csv(save_path, index=False)
        print(f"表格数据已保存至 {save_path}")


# 示例训练代码（需要在有真实数据时使用）
def train_tabular_decoder(decoder, diffusion_model, dataloader, epochs=50, lr=1e-4):
    """
    训练表格解码器
    
    Args:
        decoder: 表格解码器实例
        diffusion_model: 扩散模型实例，用于生成潜在表示
        dataloader: 包含原始表格数据的数据加载器
        epochs: 训练周期
        lr: 学习率
    """
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    
    # 损失函数：连续特征使用MSE，分类特征使用交叉熵
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    
    # 训练循环
    for epoch in range(epochs):
        total_loss = 0.0
        continuous_loss = 0.0
        categorical_loss = 0.0
        
        for batch in dataloader:
            # 假设batch包含表格数据
            if isinstance(batch, dict):
                continuous_data = batch.get('continuous')
                categorical_data = batch.get('categorical', {})
            else:
                # 处理其他数据格式
                continue
            
            # 生成潜在表示（通常由扩散模型完成）
            latent = diffusion_model.generate_latent(batch)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            continuous_pred, categorical_preds = decoder(latent)
            
            # 计算损失
            loss = 0.0
            
            # 连续特征损失
            if continuous_data is not None and continuous_pred is not None:
                c_loss = mse_loss(continuous_pred, continuous_data)
                loss += c_loss
                continuous_loss += c_loss.item()
            
            # 分类特征损失
            if categorical_data and categorical_preds:
                cat_loss = 0.0
                for feat_name, true_labels in categorical_data.items():
                    if feat_name in categorical_preds:
                        cat_loss += ce_loss(categorical_preds[feat_name], true_labels)
                
                loss += cat_loss
                categorical_loss += cat_loss.item()
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 打印进度
        print(f"Epoch [{epoch+1}/{epochs}], Total Loss: {total_loss:.4f}, "
              f"Continuous Loss: {continuous_loss:.4f}, Categorical Loss: {categorical_loss:.4f}")
    
    print("训练完成！")
    return decoder


# 示例用法
if __name__ == "__main__":
    # 创建表格解码器
    # 定义一些示例特征
    continuous_dims = 5  # 5个连续特征
    categorical_dims = {
        "category_A": 4,  # 4个类别的分类特征
        "category_B": 3,  # 3个类别的分类特征
    }
    
    decoder = TabularDecoder(
        latent_dim=16384,  # 4*64*64
        output_dims=continuous_dims,
        categorical_dims=categorical_dims,
        hidden_dims=[2048, 1024, 512]
    )
    
    # 设置特征名称（可选）
    continuous_feature_names = [f"numeric_{i}" for i in range(continuous_dims)]
    decoder.set_feature_names(continuous_feature_names=continuous_feature_names)
    
    # 设置归一化参数（可选，用于反归一化）
    feature_means = torch.zeros(continuous_dims)
    feature_stds = torch.ones(continuous_dims)
    decoder.set_normalization_params(feature_means, feature_stds)
    
    # 生成随机潜在表示
    batch_size = 5
    latent_channels = 4
    latent_height = latent_width = 64
    random_latent = torch.randn(
        batch_size, latent_channels, latent_height, latent_width,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 解码为表格数据
    decoded_df = decoder.decode(random_latent)
    print("\n解码后的表格数据示例:")
    print(decoded_df.head())
    
    # 可视化结果
    decoder.visualize_output(decoded_df)
    
    # 保存表格数据
    decoder.save_output(decoded_df)
    
    # 保存模型
    decoder.save_model()
    
    print("表格数据解码器演示完成！")
