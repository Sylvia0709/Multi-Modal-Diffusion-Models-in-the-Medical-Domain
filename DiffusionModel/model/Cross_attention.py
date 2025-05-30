# model/attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from typing import Dict

class FowardNetwork(nn.Module):
    def __init__(self, embed_dim):
        super(FowardNetwork, self).__init__()
        self.Fc1 = nn.Linear(embed_dim, embed_dim, bias=True)
        self.Fc2 = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x):
        x = F.silu(self.Fc1(x))
        x = F.silu(self.Fc2(x))
        return x


class CrossAttention(nn.Module):
    def __init__(self, latent_dim, base_modality, fusion_modality,heads):

        super(CrossAttention, self).__init__()
        self.embed_dim = latent_dim
        self.dropout = 0.2
        self.num_heads = heads

        self.current_data = base_modality
        self.fusion_data = fusion_modality

        self.batch_size = base_modality.size()[0]
        self.query_token = base_modality.size()[1]
        self.query_dimen = base_modality.size()[2]
        self.key_token = fusion_modality.size()[1]
        self.head_dim = self.embed_dim // self.num_heads

        self.W_q = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.W_k = nn.Linear(self.embed_dim,self.embed_dim, bias=False)
        self.W_v = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.O_layer = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)

        self.drop1 = nn.Dropout(self.dropout)
        self.drop2 = nn.Dropout(self.dropout)

        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.fowNet = FowardNetwork(self.embed_dim)
        self.restore= nn.Linear(self.embed_dim,self.query_dimen)

    def split_heads(self, x):
        x = x.view(self.batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def scaled_dot_product_attention(self, Q, K, V):
        scores = (torch.matmul(Q, K.transpose(-1, -2)) /
                  torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float64)))
        attn_weights = F.softmax(scores, dim=-1)
        original_mask = torch.zeros_like(attn_weights)
        mask_indices = (attn_weights >= self.alpha).float()
        natural_index = torch.arange(0, attn_weights.size(3))
        natural_index = natural_index[None,None,None,:].expand(self.batch_size,
                                                self.num_heads,attn_weights.size(2), -1)
        original_mask.scatter_(-1, natural_index, src=mask_indices)
        attn_weights = attn_weights * original_mask
        attn_weights_adjusted = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights_adjusted, V)
        return attn_output, attn_weights

    def forward(self):
        Q = self.split_heads(self.W_q(self.current_data))
        K = self.split_heads(self.W_k(self.fusion_data))
        V = self.split_heads(self.W_v(self.fusion_data))
        attn_output, atten_maps = self.scaled_dot_product_attention(Q, K, V)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(self.batch_size, self.query_token, self.embed_dim)
        attn_output = self.O_layer(attn_output)
        attn_output = self.norm1(self.current_data + self.drop1(attn_output))
        inter_output = self.fowNet(attn_output)
        final_output = self.norm2(attn_output + self.drop2(inter_output))
        final_output=self.restore(final_output)
        return final_output, atten_maps

class DynamicSequentialCrossAttention(nn.Module):
    def __init__(self, latent_dim, total_modalities: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.total_modalities = total_modalities

    def forward(self, *modalities: List[torch.Tensor]):
        """
        Args:
            modalities: List of tensors where:
                - modalities[0]: Main modality A [B, N, d_model] (Query)
                - modalities[1:]: Auxiliary modalities [B, M_i, d_model] (Keys/Values)
        Returns:
            output: [B, N, d_model] (Enhanced main modality)
        """
        assert len(modalities) == self.total_modalities, \
            f"Expected {self.total_modalities} modalities (including A), got {len(modalities)}"

        output = modalities[0]  # Initial modality[0]

        for i in range(1, self.total_modalities):
            # CrossAttention with others
            output, _ = CrossAttention(self.latent_dim, base_modality=output, fusion_modality=modalities[i], heads=4).forward()

        return output

class TokenCrossAttention(nn.Module):
    def __init__(self, latent_dim=256, num_heads=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads

    def forward(self, input_latents: List[torch.Tensor]):
        """
        input_latents: list of Tensor, each shape [N_token, latent_dim]
                       (e.g., [1, 256] for tabular, [768, 256] for image)
        Returns: fused latent of shape [latent_dim]
        """
        # Ensure all tensors are reshaped to [B, N, latent_dim]
        reshaped = []
        for x in input_latents:
            if x.dim() == 2:
                x = x.unsqueeze(0)  # [1, N, latent_dim]
            reshaped.append(x)

        model = DynamicSequentialCrossAttention(self.latent_dim, total_modalities=len(reshaped))
        output = model(*reshaped)  # [B, N, latent_dim]

        # output
        output = output.mean(dim=1)  # [B, latent_dim]
        return output.squeeze(0)     # [latent_dim]
