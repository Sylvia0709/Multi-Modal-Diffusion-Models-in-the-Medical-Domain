# model/conditioner.py
import torch
import torch.nn as nn

class EmbeddingConditioner(nn.Module):
    def __init__(self, num_modalities, num_cancer_types, latent_dim=256):
        super().__init__()
        self.task_embedding = nn.Embedding(num_modalities, latent_dim)
        self.cancer_embedding = nn.Embedding(num_cancer_types, latent_dim)
        self.condition_proj = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, task_id, cancer_id):
        """
        task_id: tensor shape [B] (e.g., [2, 0, 1])
        cancer_id: tensor shape [B]
        Returns: condition vector of shape [B, latent_dim]
        """
        task_embed = self.task_embedding(task_id)        # [B, latent_dim]
        cancer_embed = self.cancer_embedding(cancer_id)  # [B, latent_dim]
        concat = torch.cat([task_embed, cancer_embed], dim=-1)
        return self.condition_proj(concat)               # [B, latent_dim]
    

class TimestepEmbedding(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.embed = nn.Embedding(1000, latent_dim)  # assume max 1000 steps

    def forward(self, t):
        return self.embed(t)

# --- Collate Function ---
def multimodal_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    input_latents = [b['inputs'] for b in batch]
    target_tensors = [b['target'] for b in batch]  # 直接list
    target_modalities = [b['target_modality'] for b in batch]
    cancer_types = [b['cancer_type'] for b in batch]

    return {
        "inputs": input_latents,
        "target_tensor": target_tensors,
        "target_modality": target_modalities,
        "cancer_type": cancer_types
    }