import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========== Constants ==========
PATCH_FEATURE = 27
LATENT_DIM = 256
SEED = 42
ckpt_dir = "checkpoints"
timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

MODALITY_LIST = ["clinical", "mutation", "methylation", "scnv", "CT", "MR"]
modality2id = {m: i for i, m in enumerate(MODALITY_LIST)}
cancer2id = {"BLCA": 0, "KIRC": 1, "LIHC": 2}



# ========== 1. projector ==========
class ImageLatentProjector(nn.Module):
    def __init__(self,
                 input_dim: int = PATCH_FEATURE,
                 latent_dim: int = LATENT_DIM):
        super().__init__()
        self.proj = nn.Linear(input_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N_patches, 27]  ->  y: [N_patches, 256]
        """
        y = self.proj(x)  # [N,256]
        # simple encode (index -> [-1,1])
        pos = torch.arange(y.size(0), device=y.device).float().unsqueeze(1)  # [N,1]
        y = y + (pos / y.size(0) * 2 - 1)  # broadcast intp [N,256]
        return y

# ========== 2. input/target ==========
def _prep_tokens_for_training(tensor: torch.Tensor, image_projector: nn.Module) -> torch.Tensor:
    """
    modality latent to [N_token, LATENT_DIM]：
      - CT/MR: tensor.shape == [768,3,3,3] ->
          permute->reshape->projector -> [768,256]
      - omics: tensor.shape == [256] ->
          tensor.unsqueeze(0)    -> [1,256]
      - latent: [N,256]
    """
    if tensor.ndim == 4:  # CT/MR
        # (C,H,W)->(H,W,C), flatten 3×3×3=27
        t = tensor.permute(0, 2, 3, 1).reshape(-1, PATCH_FEATURE)  # [768,27]
        return image_projector(t)                                  # [768,256]
    elif tensor.ndim == 1:  # omics
        return tensor.unsqueeze(0)                                 # [1,256]
    elif tensor.ndim == 2:  # already [N,256]
        return tensor
    else:
        raise ValueError(f"Unknown latent shape {tensor.shape}")

def train_epoch(model_modules, train_loader, optimizer, scheduler, device):
    unet           = model_modules["unet"]
    fusion         = model_modules["fusion"]
    conditioner    = model_modules["conditioner"]
    t_embedder     = model_modules["t_embedder"]
    image_projector = model_modules["image_projector"]

    unet.train(); fusion.train(); conditioner.train()
    t_embedder.train(); image_projector.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc="Training"):
        if batch is None:
            continue

        inp_latents     = batch["inputs"]
        tgt_tensors     = batch["target_tensor"]
        tgt_modalities  = batch["target_modality"]
        cancer_types    = batch["cancer_type"]
        B = len(tgt_tensors)

        fused_list = []
        for i in range(B):
            tokens = [_prep_tokens_for_training(lat.to(device), image_projector) for lat in inp_latents[i]]
            fused_list.append(fusion(tokens))
        fused = torch.stack(fused_list, dim=0)

        task_ids   = torch.tensor([modality2id[m] for m in tgt_modalities], device=device)
        cancer_ids = torch.tensor([cancer2id[c]   for c in cancer_types],   device=device)
        cond_vec   = conditioner(task_ids, cancer_ids)

        t_steps = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device)
        losses = []
        for i in range(B):
            tgt_tok = _prep_tokens_for_training(tgt_tensors[i].to(device), image_projector)
            noise   = torch.randn_like(tgt_tok)
            z_t     = scheduler.add_noise(tgt_tok, noise, t_steps[i]).unsqueeze(0)
            t_emb   = t_embedder(t_steps[i].unsqueeze(0))
            pred    = unet(z_t, fused[i].unsqueeze(0), cond_vec[i].unsqueeze(0), t_emb)
            losses.append(F.mse_loss(pred.squeeze(0), noise))
        batch_loss = torch.stack(losses).mean()

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()

    return total_loss / len(train_loader)

def val_epoch(model_modules, val_loader, scheduler, device):
    unet           = model_modules["unet"]
    fusion         = model_modules["fusion"]
    conditioner    = model_modules["conditioner"]
    t_embedder     = model_modules["t_embedder"]
    image_projector = model_modules["image_projector"]

    unet.eval(); fusion.eval(); conditioner.eval()
    t_embedder.eval(); image_projector.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            if batch is None:
                continue

            inp_latents     = batch["inputs"]
            tgt_tensors     = batch["target_tensor"]
            tgt_modalities  = batch["target_modality"]
            cancer_types    = batch["cancer_type"]
            B = len(tgt_tensors)

            fused_list = []
            for i in range(B):
                tokens = [_prep_tokens_for_training(lat.to(device), image_projector) for lat in inp_latents[i]]
                fused_list.append(fusion(tokens))
            fused = torch.stack(fused_list, dim=0)

            task_ids   = torch.tensor([modality2id[m] for m in tgt_modalities], device=device)
            cancer_ids = torch.tensor([cancer2id[c]   for c in cancer_types],   device=device)
            cond_vec   = conditioner(task_ids, cancer_ids)

            t_steps = torch.randint(0, scheduler.config.num_train_timesteps, (B,), device=device)
            losses = []
            for i in range(B):
                tgt_tok = _prep_tokens_for_training(tgt_tensors[i].to(device), image_projector)
                noise   = torch.randn_like(tgt_tok)
                z_t     = scheduler.add_noise(tgt_tok, noise, t_steps[i]).unsqueeze(0)
                t_emb   = t_embedder(t_steps[i].unsqueeze(0))
                pred    = unet(z_t, fused[i].unsqueeze(0), cond_vec[i].unsqueeze(0), t_emb)
                losses.append(F.mse_loss(pred.squeeze(0), noise))
            total_loss += torch.stack(losses).mean().item()

    return total_loss / len(val_loader)

def train_model(model_modules, train_loader, val_loader, optimizer, scheduler, device, num_epochs=30, patience=5):
    
    unet           = model_modules["unet"]
    fusion         = model_modules["fusion"]
    conditioner    = model_modules["conditioner"]
    t_embedder     = model_modules["t_embedder"]
    image_projector = model_modules["image_projector"]
    
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []
    best_model_state = None
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n[Epoch {epoch}]")
        train_loss = train_epoch(model_modules, train_loader, optimizer, scheduler, device)
        val_loss   = val_epoch(model_modules, val_loader, scheduler, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch} | Train {train_loss:.4f} | Val {val_loss:.4f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {
                "unet":             unet.state_dict(),
                "fusion":           fusion.state_dict(),
                "conditioner":      conditioner.state_dict(),
                "t_embedder":       t_embedder.state_dict(),
                "image_projector":  image_projector.state_dict(),
                "epoch":            epoch,
                "val_loss":         val_loss,
            }
            ckpt_path = os.path.join(
                ckpt_dir,
                f"best_seed{SEED}_{timestamp_tag}_epoch{epoch}.pt"
            )
            torch.save(best_model_state, ckpt_path)
            print("Saved best model to", ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stop triggered.")
                break

    # Plot
    epochs = range(1, len(train_losses) + 1)
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = val_losses[best_epoch - 1]

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.scatter(best_epoch, best_val_loss, color='red', marker='*', s=200, label=f'Best Epoch {best_epoch}')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend(); plt.grid(True)
    plt.savefig(f"loss_curve_seed{SEED}.png")
    plt.show()

    return train_losses, val_losses, best_model_state

def _prep_tokens_eval(tensor: torch.Tensor, image_projector: nn.Module) -> torch.Tensor:
    """omics →[1,256]   |   CT/MR →[768,256]"""
    if tensor.ndim == 4:                                # CT / MR
        t = tensor.permute(0,2,3,1).reshape(-1, 27)     # [768,27]
        return image_projector(t)                       # [768,256]
    elif tensor.ndim == 1:                              # omics
        return tensor.unsqueeze(0)                      # [1,256]
    else:
        return tensor

@torch.no_grad()
def evaluate_on_test(model_modules,
                     scheduler,
                     test_patient_dict: dict,
                     modality_to_latent: dict,
                     image_latents: dict,
                     batch_size: int = 16,
                     device: str = "cuda"):

    unet           = model_modules["unet"]
    fusion         = model_modules["fusion"]
    conditioner    = model_modules["conditioner"]
    t_embedder     = model_modules["t_embedder"]
    image_projector = model_modules["image_projector"]

    all_mse, mse_by_mod = [], {m: [] for m in
                               ["mutation","methylation","scnv","CT","MR"]}

    patient_ids = list(test_patient_dict.keys())

    for pid in tqdm(patient_ids, desc="test-patients"):

        info  = test_patient_dict[pid]
        avail = info["modalities"]; cancer = info["cancer_type"]

        # non-clinical as target
        targets = [m for m in avail if m!="clinical"]
        if not targets: continue
        tgt = random.choice(targets)

        # -------- inputs --------
        inputs = {}
        for m in avail:
            if m == tgt: continue
            if m in ("CT","MR"):
                if pid in image_latents and m in image_latents[pid]:
                    inputs[m] = image_latents[pid][m]
            else:
                try:
                    inputs[m] = torch.tensor(
                        modality_to_latent[m].loc[pid].values,
                        dtype=torch.float32)
                except KeyError:
                    pass
        if not inputs: continue

        # -------- sample --------
        gen_lat = sample_target_latent(
            unet, fusion, conditioner, t_embedder, scheduler,
            image_projector,
            patient_inputs=inputs,
            cancer_type=cancer,
            target_modality=tgt,
            latent_dim=256,
            device=device)

        # -------- GT latent --------
        if tgt in ("CT","MR"):
            if pid in image_latents and tgt in image_latents[pid]:
                true_lat = _prep_tokens_eval(image_latents[pid][tgt], image_projector).to(device)
            else:
                continue
        else:
            try:
                v = modality_to_latent[tgt].loc[pid].values
                true_lat = torch.tensor(v, dtype=torch.float32,
                                        device=device).unsqueeze(0)
            except KeyError:
                continue

        # -------- MSE --------
        mse = F.mse_loss(gen_lat, true_lat).item()
        all_mse.append(mse); mse_by_mod[tgt].append(mse)

    # -------- report --------
    if all_mse:
        print(f"\nOverall   MSE = {np.mean(all_mse):.5f}")
    for m, lst in mse_by_mod.items():
        if lst:
            print(f"{m:<11} MSE = {np.mean(lst):.5f}  (n={len(lst)})")
        else:
            print(f"{m:<11} —  no sample")


# Sampling with DDPM Scheduler
def _prep_tokens_sampling(tensor, projector):
    """
    Omics  [256]        -> [1,256]
    CT/MR  [768,3,3,3]  -> [768,27] -> projector -> [768,256]
    """
    if tensor.ndim == 4:                           # CT/MR
        t = tensor.permute(0, 2, 3, 1).reshape(-1, 27)
        return projector(t)                        # [768,256]
    elif tensor.ndim == 1:                         # omics
        return tensor.unsqueeze(0)                 # [1,256]
    elif tensor.ndim == 2:                         # already [N,256]
        return tensor
    else:
        raise ValueError(f"Unknown latent shape {tensor.shape}")

@torch.no_grad()
def sample_target_latent(
        unet, fusion, conditioner, t_embedder, scheduler,
        image_projector,
        patient_inputs: dict,           # { modality : tensor }
        cancer_type: str,
        target_modality: str,
        latent_dim: int = 256,
        device: str = "cuda") -> torch.Tensor:
    """
    based on inputs (omics &/or image latent) reverse sample target latent。
    output：
        • omics : [1,256]
        • CT/MR : [768,256]
    """

    # ---------- 1. fuse ----------
    tokens = [ _prep_tokens_sampling(lat.to(device), image_projector)
               for lat in patient_inputs.values() ]
    fused  = fusion(tokens).unsqueeze(0)                   # [1,256]

    # ---------- 2. condition ----------
    task   = torch.tensor([modality2id[target_modality]], device=device)
    cancer = torch.tensor([cancer2id[cancer_type]],       device=device)
    cond   = conditioner(task, cancer)                    # [1,256]

    # ---------- 3. noise ----------
    N_token = 768 if target_modality in ("CT", "MR") else 1
    x = torch.randn((1, N_token, latent_dim), device=device)  # [1,N,256]

    # ---------- 4. back prop ----------
    for t in scheduler.timesteps:
        t_emb = t_embedder(torch.tensor([t], device=device))      # [1,256]
        pred  = unet(x, fused, cond, t_emb)                       # [1,N,256]
        x     = scheduler.step(pred, t, x).prev_sample            # [1,N,256]

    return x.squeeze(0)                                           # [N,256]


# Save latent vectors
def denormalize_latent(latent: torch.Tensor, modality: str, stats: dict):
    """
    Denormalization latent
    latent:     torch.Tensor，shape [N, latent_dim] 或 [1,latent_dim]
    modality:   str, 如 "mutation","CT" 等
    stats:      dict, {modality: (mu, sigma)}
    """
    mu, sigma = stats[modality]
    return latent * sigma + mu

@torch.no_grad()
def generate_and_save_latents(
    unet, fusion, conditioner, t_embedder, scheduler,
    patient_dict, modality_to_latent, image_latents,
    stats,               # {modality: (mu, sigma)}
    save_dir,
    latent_dim=LATENT_DIM,
    device="cuda"
):
    unet.eval()
    fusion.eval()
    conditioner.eval()
    t_embedder.eval()
    image_projector.eval()

    os.makedirs(save_dir, exist_ok=True)
    save_count = 0

    for pid, info in tqdm(patient_dict.items(), desc="gen-latents"):
        cancer = info["cancer_type"]
        task_list = info.get("tasks", [])

        for task in task_list:
            input_modalities = task["input_modalities"]
            target_modality  = task["target_modality"]

            # ---- 1. prepare inputs ----
            inputs = {}
            for m in input_modalities:
                if m in ("CT", "MR") and pid in image_latents and m in image_latents[pid]:
                    raw = image_latents[pid][m]  # Tensor [768,3,3,3]
                    tmp = raw.permute(0, 2, 3, 1).reshape(-1, PATCH_FEATURE)  # [768,27]
                    proj = image_projector(tmp.to(device))  # [768,256]
                    inputs[m] = proj
                elif m in modality_to_latent:
                    vec = modality_to_latent[m].loc[pid].values.astype(np.float32)
                    inputs[m] = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)

            if not inputs:
                continue


            gen = sample_target_latent(
                unet=unet,
                fusion=fusion,
                conditioner=conditioner,
                t_embedder=t_embedder,
                scheduler=scheduler,
                patient_inputs=inputs,
                cancer_type=cancer,
                target_modality=target_modality,
                latent_dim=latent_dim,
                device=device
            )

            gen_denorm = denormalize_latent(gen, target_modality, stats)

            input_mods = sorted(input_modalities)
            fn = f"{pid}_{cancer}_{target_modality}_{'+'.join(input_mods)}.pt"
            torch.save(gen_denorm.cpu(), os.path.join(save_dir, fn))
            save_count += 1

    print(f"All done! Total saved latent files: {save_count}")
