# ConditionedUNet
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionedUNet(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim

        self.fc_in = nn.Linear(latent_dim, latent_dim)

        self.film_gamma = nn.Linear(latent_dim * 2, latent_dim)
        self.film_beta = nn.Linear(latent_dim * 2, latent_dim)
        self.film_norm = nn.LayerNorm(latent_dim)

        self.fc_mid = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Dropout(0.1)
        )

        self.fc_out = nn.Linear(latent_dim, latent_dim)

        # initial to 0
        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

    def forward(self, z_t, fused_latent, condition, t_embed):
        """
        z_t: Tensor
            - [B, latent_dim] (tabular, no spatial info) 或
            - [B, N_token, latent_dim] (spatial, e.g., MR/CT)
        fused_latent: [B, latent_dim]
        condition: [B, latent_dim]
        t_embed: [B, latent_dim]
        """
        # unify z_t dimension -> [B, N_token, latent_dim]
        if z_t.ndim == 2:
            # [B, latent_dim] -> [B, 1, latent_dim]
            z_t = z_t.unsqueeze(1)
        elif z_t.ndim != 3:
            raise ValueError(f"Invalid z_t shape: {z_t.shape}, expected [B, latent_dim] or [B, N_token, latent_dim]")

        B, N, D = z_t.shape  

        # Expand fused latent → [B, N_token, latent_dim]
        fused_latent_expanded = fused_latent.unsqueeze(1).expand(-1, N, -1)

        # input
        x = self.fc_in(z_t + fused_latent_expanded)

        # condition
        cond_input = torch.cat([condition, t_embed], dim=-1)  # [B, latent_dim * 2]
        gamma = self.film_gamma(cond_input).unsqueeze(1)      # [B, 1, latent_dim]
        beta = self.film_beta(cond_input).unsqueeze(1)        # [B, 1, latent_dim]

        # main
        h = self.fc_mid(x)
        h = h * (1 + gamma) + beta  # FiLM modulation
        h = self.film_norm(h)

        out = self.fc_out(F.silu(h))

        return out + x  # Residual