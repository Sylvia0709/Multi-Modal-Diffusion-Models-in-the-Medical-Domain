import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import pickle
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import umap


class OmicsDecoder(nn.Module):
    """
    组学数据解码器 - 将潜在表示(latent)解码为组学数据
    用于多模态扩散模型框架的组学数据模态解码部分
    
    支持多种组学数据类型:
    - 基因表达数据（RNA-seq/microarray）
    - DNA甲基化数据
    - 蛋白组数据
    """
    
    def __init__(self, latent_dim=16384, # 4*64*64 
                 omics_dim=10000,
                 omics_type="gene_expression",
                 use_attention=True,
                 gene_embeddings=None,
                 hidden_dims=[2048, 1024, 512],
                 dropout_rate=0.3,
                 activation=nn.LeakyReLU(0.2),
                 device=None,
                 output_dir="output/omics"):
        """
        初始化组学数据解码器
        
        Args:
            latent_dim: 潜在表示的维度（对于SD通常是4*64*64=16384）
            omics_dim: 组学数据维度（例如基因数量）
            omics_type: 组学数据类型，支持"gene_expression", "methylation", "proteomics"
            use_attention: 是否使用注意力机制
            gene_embeddings: 预训练的基因嵌入 (形状应为 [gene_count, embedding_dim])
            hidden_dims: 隐藏层维度的列表
            dropout_rate: Dropout比率
            activation: 激活函数
            device: 计算设备，如果为None则自动检测
            output_dir: 输出目录
        """
        super(OmicsDecoder, self).__init__()
        
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # 创建输出目录
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # 配置
        self.latent_dim = latent_dim
        self.omics_dim = omics_dim
        self.omics_type = omics_type
        self.use_attention = use_attention
        
        # 记录特征名称
        self.feature_names = None
        
        # 记录归一化参数
        self.feature_means = None
        self.feature_stds = None
        
        # 展平层
        self.flatten = nn.Flatten()
        
        # 主干网络
        backbone_layers = []
        in_dim = latent_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            backbone_layers.append(nn.Linear(in_dim, hidden_dim))
            backbone_layers.append(activation)
            backbone_layers.append(nn.BatchNorm1d(hidden_dim))
            backbone_layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim
        
        self.backbone = nn.Sequential(*backbone_layers)
        
        # 注意力机制
        if use_attention:
            self.attention_dim = hidden_dims[-1]
            self.multihead_attn = nn.MultiheadAttention(
                embed_dim=self.attention_dim,
                num_heads=8,
                batch_first=True
            )
            
            # 使用基因嵌入（如果提供）
            if gene_embeddings is not None:
                # 确保基因嵌入是tensor
                if not isinstance(gene_embeddings, torch.Tensor):
                    gene_embeddings = torch.tensor(gene_embeddings, dtype=torch.float32)
                
                # 调整嵌入维度
                if gene_embeddings.shape[1] != self.attention_dim:
                    self.gene_embedding_projector = nn.Linear(gene_embeddings.shape[1], self.attention_dim)
                    self.gene_embeddings = self.gene_embedding_projector(gene_embeddings)
                else:
                    self.gene_embeddings = gene_embeddings
                
                # 移到设备上
                self.gene_embeddings = self.gene_embeddings.to(self.device)
                
                # 设置为不需要梯度
                self.gene_embeddings.requires_grad_(False)
                
                print(f"使用预训练基因嵌入，形状: {self.gene_embeddings.shape}")
        
        # 输出层 - 根据组学数据类型选择适当的激活函数
        final_activation = None
        if omics_type == "gene_expression":
            # 基因表达通常为非负值
            final_activation = nn.Softplus()
        elif omics_type == "methylation":
            # 甲基化为0-1之间的值
            final_activation = nn.Sigmoid()
        elif omics_type == "proteomics":
            # 蛋白质表达通常为非负值
            final_activation = nn.ReLU()
        
        # 构建输出层
        output_layers = []
        output_layers.append(nn.Linear(hidden_dims[-1], omics_dim))
        if final_activation is not None:
            output_layers.append(final_activation)
        
        self.output_layer = nn.Sequential(*output_layers)
        
        # 移动到设备
        self.to(self.device)
        print(f"组学数据解码器已初始化，类型: {omics_type}, 设备: {self.device}")
        print(f"输出维度: {omics_dim}, 使用注意力: {use_attention}")
    
    def forward(self, latent):
        """
        前向传播
        
        Args:
            latent: 潜在表示，形状为[B, C, H, W]
            
        Returns:
            output: 组学数据输出
        """
        # 确保输入是tensor
        if not isinstance(latent, torch.Tensor):
            latent = torch.tensor(latent, device=self.device)
        
        # 确保形状正确
        if latent.dim() == 3:
            latent = latent.unsqueeze(0)  # 添加批次维度
        
        # 移动到正确的设备
        latent = latent.to(self.device)
        
        # 展平潜在表示
        x = self.flatten(latent)
        
        # 通过主干网络
        features = self.backbone(x)
        
        # 应用注意力（如果启用）
        if self.use_attention and hasattr(self, 'multihead_attn'):
            # 准备查询
            query = features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            # 如果有基因嵌入，使用它作为键和值
            if hasattr(self, 'gene_embeddings'):
                # 扩展基因嵌入以匹配批次大小
                batch_size = latent.shape[0]
                key_value = self.gene_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
                
                # 应用注意力
                attn_output, _ = self.multihead_attn(query, key_value, key_value)
                features = attn_output.squeeze(1)  # [batch_size, hidden_dim]
        
        # 生成组学数据输出
        output = self.output_layer(features)
        
        return output
    
    def decode(self, latent, denormalize=True):
        """
        将潜在表示解码为组学数据
        
        Args:
            latent: 潜在表示
            denormalize: 是否反归一化
            
        Returns:
            df: pandas DataFrame包含解码后的组学数据
        """
        with torch.no_grad():
            omics_output = self.forward(latent)
        
        # 转换为numpy
        omics_np = omics_output.cpu().numpy()
        
        # 反归一化（如果有归一化参数且需要反归一化）
        if denormalize and self.feature_means is not None and self.feature_stds is not None:
            omics_np = omics_np * self.feature_stds + self.feature_means
        
        # 创建DataFrame
        feature_names = self.feature_names
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.omics_dim)]
        
        # 确保特征名称数量与输出维度匹配
        if len(feature_names) != self.omics_dim:
            feature_names = feature_names[:self.omics_dim] if len(feature_names) > self.omics_dim else \
                            feature_names + [f"feature_{i}" for i in range(len(feature_names), self.omics_dim)]
        
        df = pd.DataFrame(omics_np, columns=feature_names)
        
        return df
    
    def set_feature_names(self, feature_names):
        """
        设置特征名称
        
        Args:
            feature_names: 特征名称列表
        """
        if len(feature_names) != self.omics_dim:
            print(f"警告: 特征名称数量({len(feature_names)})与输出维度({self.omics_dim})不匹配")
        
        self.feature_names = feature_names
        print(f"已设置{len(feature_names)}个特征名称")
    
    def set_normalization_params(self, feature_means, feature_stds):
        """
        设置特征归一化参数，用于反归一化
        
        Args:
            feature_means: 特征均值
            feature_stds: 特征标准差
        """
        self.feature_means = feature_means
        self.feature_stds = feature_stds
        print("已设置归一化参数，解码时将自动反归一化")
    
    def visualize_output(self, df, n_features=50, methods=["pca", "umap"], save=True, filename_prefix="omics_visualization"):
        """
        可视化解码后的组学数据
        
        Args:
            df: 解码后的DataFrame
            n_features: 用于可视化的特征数量
            methods: 可视化方法列表，支持"pca"、"umap"、"heatmap"和"distribution"
            save: 是否保存可视化结果
            filename_prefix: 保存文件名前缀
        """
        # 确保methods是列表
        if isinstance(methods, str):
            methods = [methods]
        
        # 仅使用前n_features个特征进行可视化
        if df.shape[1] > n_features:
            df_subset = df.iloc[:, :n_features]
            print(f"使用前{n_features}个特征进行可视化")
        else:
            df_subset = df
        
        for method in methods:
            if method.lower() == "pca":
                self._visualize_pca(df, save, f"{filename_prefix}_pca.png")
            elif method.lower() == "umap":
                self._visualize_umap(df, save, f"{filename_prefix}_umap.png")
            elif method.lower() == "heatmap":
                self._visualize_heatmap(df_subset, save, f"{filename_prefix}_heatmap.png")
            elif method.lower() == "distribution":
                self._visualize_distributions(df_subset, save, f"{filename_prefix}_distributions.png")
            else:
                print(f"不支持的可视化方法: {method}")
    
    def _visualize_pca(self, df, save=True, filename="omics_pca.png"):
        """PCA可视化"""
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df.values)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
        plt.title(f"组学数据PCA可视化 ({self.omics_type})")
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
        plt.grid(alpha=0.3)
        
        if save:
            save_path = self.output_dir / filename
            plt.savefig(save_path)
            print(f"PCA可视化已保存至 {save_path}")
        
        plt.show()
    
    def _visualize_umap(self, df, save=True, filename="omics_umap.png"):
        """UMAP可视化"""
        try:
            reducer = umap.UMAP()
            umap_result = reducer.fit_transform(df.values)
            
            plt.figure(figsize=(10, 8))
            plt.scatter(umap_result[:, 0], umap_result[:, 1], alpha=0.7)
            plt.title(f"组学数据UMAP可视化 ({self.omics_type})")
            plt.xlabel("UMAP1")
            plt.ylabel("UMAP2")
            plt.grid(alpha=0.3)
            
            if save:
                save_path = self.output_dir / filename
                plt.savefig(save_path)
                print(f"UMAP可视化已保存至 {save_path}")
            
            plt.show()
        except ImportError:
            print("UMAP可视化需要安装umap-learn库: pip install umap-learn")
    
    def _visualize_heatmap(self, df, save=True, filename="omics_heatmap.png"):
        """热图可视化"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.T, cmap="viridis")
        plt.title(f"组学数据热图 ({self.omics_type})")
        plt.xlabel("样本")
        plt.ylabel("特征")
        
        if save:
            save_path = self.output_dir / filename
            plt.savefig(save_path)
            print(f"热图已保存至 {save_path}")
        
        plt.show()
    
    def _visualize_distributions(self, df, save=True, filename="omics_distributions.png"):
        """分布可视化"""
        # 计算均值和标准差
        means = df.mean(axis=0)
        stds = df.std(axis=0)
        
        plt.figure(figsize=(12, 6))
        
        # 绘制均值分布
        plt.subplot(1, 2, 1)
        plt.hist(means, bins=30, alpha=0.7)
        plt.title(f"特征均值分布 ({self.omics_type})")
        plt.xlabel("均值")
        plt.ylabel("频率")
        plt.grid(alpha=0.3)
        
        # 绘制标准差分布
        plt.subplot(1, 2, 2)
        plt.hist(stds, bins=30, alpha=0.7)
        plt.title(f"特征标准差分布 ({self.omics_type})")
        plt.xlabel("标准差")
        plt.ylabel("频率")
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.output_dir / filename
            plt.savefig(save_path)
            print(f"分布可视化已保存至 {save_path}")
        
        plt.show()
    
    def analyze_biological_coherence(self, df, reference_df=None):
        """
        分析生物学一致性（如果可用参考数据）
        
        Args:
            df: 解码后的DataFrame
            reference_df: 参考组学数据DataFrame
            
        Returns:
            results: 分析结果字典
        """
        results = {}
        
        # 如果有参考数据，计算相关性
        if reference_df is not None:
            # 确保两个数据框具有相同的列
            common_cols = list(set(df.columns) & set(reference_df.columns))
            if len(common_cols) == 0:
                print("警告: 解码数据与参考数据没有共同的特征")
                return results
            
            print(f"使用{len(common_cols)}个共同特征进行分析")
            
            # 计算每个样本的相关性
            correlations = []
            for i in range(len(df)):
                if i < len(reference_df):
                    corr, _ = pearsonr(
                        df.loc[i, common_cols].values,
                        reference_df.loc[i, common_cols].values
                    )
                    correlations.append(corr)
            
            results['correlations'] = correlations
            results['mean_correlation'] = np.mean(correlations)
            results['median_correlation'] = np.median(correlations)
            
            print(f"与参考数据的平均相关性: {results['mean_correlation']:.4f}")
            print(f"与参考数据的中位相关性: {results['median_correlation']:.4f}")
            
            # 绘制相关性分布
            plt.figure(figsize=(8, 6))
            plt.hist(correlations, bins=20, alpha=0.7)
            plt.title("与参考数据的样本相关性分布")
            plt.xlabel("Pearson相关系数")
            plt.ylabel("样本数")
            plt.grid(alpha=0.3)
            plt.axvline(results['mean_correlation'], color='r', linestyle='--', label=f"平均值: {results['mean_correlation']:.4f}")
            plt.legend()
            plt.savefig(self.output_dir / "correlation_distribution.png")
            plt.show()
        
        return results
    
    def save_output(self, df, filename="decoded_omics_data.csv"):
        """
        保存解码后的组学数据
        
        Args:
            df: 解码后的DataFrame
            filename: 保存的文件名
        """
        save_path = self.output_dir / filename
        df.to_csv(save_path, index=False)
        print(f"组学数据已保存至 {save_path}")
    
    def save_model(self, filename="omics_decoder.pt"):
        """
        保存模型和配置
        
        Args:
            filename: 保存的文件名
        """
        save_path = self.output_dir / filename
        
        model_state = {
            'model_state_dict': self.state_dict(),
            'latent_dim': self.latent_dim,
            'omics_dim': self.omics_dim,
            'omics_type': self.omics_type,
            'use_attention': self.use_attention,
            'feature_names': self.feature_names,
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds
        }
        
        torch.save(model_state, save_path)
        print(f"模型已保存至 {save_path}")
    
    @classmethod
    def load_model(cls, filename, device=None, output_dir="output/omics"):
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
            omics_dim=model_state['omics_dim'],
            omics_type=model_state['omics_type'],
            use_attention=model_state['use_attention'],
            device=device,
            output_dir=output_dir
        )
        
        model.load_state_dict(model_state['model_state_dict'])
        model.feature_names = model_state['feature_names']
        model.feature_means = model_state['feature_means']
        model.feature_stds = model_state['feature_stds']
        
        print(f"模型已从 {model_path} 加载")
        return model


# 示例训练代码
def train_omics_decoder(decoder, diffusion_model, dataloader, epochs=50, lr=1e-4):
    """
    训练组学数据解码器
    
    Args:
        decoder: 组学数据解码器实例
        diffusion_model: 扩散模型实例，用于生成潜在表示
        dataloader: 包含原始组学数据的数据加载器
        epochs: 训练周期
        lr: 学习率
    """
    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    
    # 损失函数：基于组学数据类型选择合适的损失函数
    if decoder.omics_type == "gene_expression":
        # 基因表达可以使用MSE或Poisson损失
        criterion = nn.MSELoss()
    elif decoder.omics_type == "methylation":
        # 甲基化可以使用二元交叉熵
        criterion = nn.BCELoss()
    else:
        # 默认使用MSE
        criterion = nn.MSELoss()
    
    # 训练循环
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch in dataloader:
            # 假设batch包含组学数据
            omics_data = batch
            
            # 生成潜在表示（通常由扩散模型完成）
            latent = diffusion_model.generate_latent(batch)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            pred = decoder(latent)
            
            # 计算损失
            loss = criterion(pred, omics_data)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 打印进度
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.4f}")
    
    print("训练完成！")
    return decoder


# 示例用法
if __name__ == "__main__":
    # 创建组学数据解码器
    omics_dim = 5000  # 例如5000个基因
    decoder = OmicsDecoder(
        latent_dim=16384,  # 4*64*64
        omics_dim=omics_dim,
        omics_type="gene_expression",
        use_attention=True,
        hidden_dims=[2048, 1024, 512]
    )
    
    # 设置特征名称（通常是基因名称）
    feature_names = [f"gene_{i}" for i in range(omics_dim)]
    decoder.set_feature_names(feature_names)
    
    # 设置归一化参数
    feature_means = np.zeros(omics_dim)
    feature_stds = np.ones(omics_dim)
    decoder.set_normalization_params(feature_means, feature_stds)
    
    # 生成随机潜在表示（通常从扩散模型获得）
    batch_size = 10
    latent_channels = 4
    latent_height = latent_width = 64
    random_latent = torch.randn(
        batch_size, latent_channels, latent_height, latent_width,
        device=decoder.device
    )
    
    # 解码为组学数据
    decoded_df = decoder.decode(random_latent)
    print("\n解码后的组学数据示例:")
    print(decoded_df.iloc[:5, :10])  # 显示前5个样本的前10个特征
    
    # 可视化结果
    decoder.visualize_output(decoded_df, methods=["pca", "heatmap", "distribution"])
    
    # 保存组学数据
    decoder.save_output(decoded_df)
    
    # 保存模型
    decoder.save_model()
    
    print("组学数据解码器演示完成！")
