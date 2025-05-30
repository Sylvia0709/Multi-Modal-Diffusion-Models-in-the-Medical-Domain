# model/attention.py

import torch
import torch.nn as nn

class TokenCrossAttention(nn.Module):
    def __init__(self, latent_dim=256, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, latent_dim))  # Learnable query

    def forward(self, input_latents):
        """
        input_latents: list of Tensor, each shape [N_token, latent_dim]
                       (N_token = 1 for tabular latent, ~768 for image latent)
        Returns: fused latent of shape [latent_dim]
        """
        x = torch.cat(input_latents, dim=0).unsqueeze(0)  # [1, total_tokens, latent_dim]

        query = self.query.expand(1, 1, -1)  # [1, 1, latent_dim]

        fused, _ = self.attn(query, x, x)  # [1, 1, latent_dim]

        return fused.squeeze(0).squeeze(0)  # [latent_dim]
