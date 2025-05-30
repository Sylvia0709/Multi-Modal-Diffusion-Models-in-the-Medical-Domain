# 1. Import libraries + config
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from datetime import datetime
import random
import os

from diffusers import DDPMScheduler

from data_loader import load_data, MultiModalTranslationDataset, split_data
from model.Cross_attention_Token_Only import TokenCrossAttention
from model.conditioner import EmbeddingConditioner, TimestepEmbedding, multimodal_collate_fn
from model.unet import ConditionedUNet
from train_generate import train_model, evaluate_on_test, generate_and_save_latents, ImageLatentProjector

try:
    import monai
except ImportError:
    get_ipython().system('pip install -q monai')
    import monai

# ========== 0. config ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
PATCH_FEATURE = 27           # 3×3×3 patch flattened
LATENT_DIM    = 256
timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
ckpt_dir      = "checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

# === set seed ===
SEED = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

# modality & cancer maps
MODALITY_LIST = ["clinical", "mutation", "methylation", "scnv", "CT", "MR"]
modality2id = {m: i for i, m in enumerate(MODALITY_LIST)}

# Fill this when you know all types
cancer2id = {"BLCA": 0, "KIRC": 1, "LIHC": 2}

def main():
    from google.colab import drive
    drive.mount('/content/drive')
    
    omic_path = "/content/drive/MyDrive/proj72_data/multiomics/encoded_omics_no_group/"
    task_path = "/content/drive/MyDrive/proj72_data/train_test_split"
    image_path = "/content/drive/MyDrive/proj72_data/"
    
    clinical = pd.read_csv(os.path.join(omic_path, "clinical_features.csv"), index_col=0).drop(columns=["cancer_type"])
    mutation = pd.read_csv(os.path.join(omic_path, "mutation_features.csv"), index_col=0).drop(columns=["cancer_type"])
    methylation = pd.read_csv(os.path.join(omic_path, "methylation_features.csv"), index_col=0).drop(columns=["cancer_type"])
    scnv = pd.read_csv(os.path.join(omic_path, "scnv_features.csv"), index_col=0).drop(columns=["cancer_type"])

    patient_dict, image_latents, stats, multimodal_tasks = load_data(omic_path, image_path, task_path, MODALITY_LIST)

    # split data into train and val
    train_patients, val_patients, train_patient_dict, val_patient_dict = split_data(patient_dict)
    train_dataset = MultiModalTranslationDataset(task_table=multimodal_tasks,
                                             patient_dict=train_patient_dict,
                                             split="train")

    val_dataset = MultiModalTranslationDataset(task_table=multimodal_tasks,
                                               patient_dict=val_patient_dict,
                                               split="train")  
    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=multimodal_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=multimodal_collate_fn)

    # ========== 1. projector ==========
    image_projector = ImageLatentProjector().to(device)

    # ========== 2. input/target ==========
    unet = ConditionedUNet(latent_dim=LATENT_DIM)
    fusion = TokenCrossAttention(latent_dim=LATENT_DIM)
    conditioner = EmbeddingConditioner(
        num_modalities=len(modality2id),
        num_cancer_types=len(cancer2id),
        latent_dim=LATENT_DIM
    )
    t_embedder = TimestepEmbedding(latent_dim=LATENT_DIM)
    image_projector = ImageLatentProjector(
        input_dim=PATCH_FEATURE,
        latent_dim=LATENT_DIM
    )
    model_modules = {
        "unet": unet.to(device),
        "fusion": fusion.to(device),
        "conditioner": conditioner.to(device),
        "t_embedder": t_embedder.to(device),
        "image_projector": image_projector.to(device)
    }
    optimizer = torch.optim.AdamW(
        list(unet.parameters()) +
        list(fusion.parameters()) +
        list(conditioner.parameters()) +
        list(t_embedder.parameters()) +
        list(image_projector.parameters()),
        lr=3e-4, weight_decay=1e-2
    )

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=1e-4, beta_end=2e-2,
        beta_schedule="linear", prediction_type="epsilon"
    )
    scheduler.set_timesteps(1000)

    num_epochs, patience = 50, 5

    # ============ Training ============
    train_losses, val_losses, best_model_state = train_model(
        model_modules=model_modules, 
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        patience=5,
        device=device)


    # ============  Re-build modules & load ckpt ============
    unet         = ConditionedUNet(latent_dim=256).to(device)
    fusion       = TokenCrossAttention(latent_dim=256).to(device)
    conditioner  = EmbeddingConditioner(len(modality2id), len(cancer2id),
                                        latent_dim=256).to(device)
    t_embedder   = TimestepEmbedding(latent_dim=256).to(device)
    image_projector = ImageLatentProjector(input_dim=27,
                                           latent_dim=256).to(device)
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=1e-4, beta_end=2e-2,
        beta_schedule="linear", prediction_type="epsilon"
    )
    scheduler.set_timesteps(1000)


    unet.load_state_dict(best_model_state["unet"])
    fusion.load_state_dict(best_model_state["fusion"])
    conditioner.load_state_dict(best_model_state["conditioner"])
    t_embedder.load_state_dict(best_model_state["t_embedder"])
    image_projector.load_state_dict(best_model_state["image_projector"])

    unet.eval(); fusion.eval(); conditioner.eval()
    t_embedder.eval(); image_projector.eval()

    model_modules = {
        "unet": unet.to(device),
        "fusion": fusion.to(device),
        "conditioner": conditioner.to(device),
        "t_embedder": t_embedder.to(device),
        "image_projector": image_projector.to(device)
    }

    print("modules rebuilt & weights loaded.")

    # Prepare for test set
    test_patients = [p for p in patient_dict if patient_dict[p]["split"] == "test"]

    test_patient_dict = {p: patient_dict[p] for p in test_patients}
    test_dataset = MultiModalTranslationDataset(task_table=multimodal_tasks,
                                                patient_dict=test_patient_dict,
                                                split='test')

    # Test DataLoader
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=multimodal_collate_fn)

    # ============ evaluation on test ============
    evaluate_on_test(model_modules,
                     scheduler,
                     test_patient_dict,
                     modality_to_latent={"mutation": mutation,
                                         "methylation": methylation,
                                         "scnv": scnv},
                     image_latents=image_latents,
                     batch_size=16,
                     device=device)
    # ============ save latent vectors ============

    # train patient
    train_patients_all     = [p for p in patient_dict if patient_dict[p]["split"] == "train"]
    train_patient_dict_all = {p: patient_dict[p] for p in train_patients_all}

    modality_to_latent = {
        "mutation": mutation,
        "methylation": methylation,
        "scnv": scnv,
        "clinical": clinical
    }
    base_path = "/content/drive/MyDrive/proj72_data/Diffusion_model"
    save_dir = os.path.join(base_path, "new_generated_train_latents")
    os.makedirs(save_dir, exist_ok=True)

    generate_and_save_latents(
        unet=unet,
        fusion=fusion,
        conditioner=conditioner,
        t_embedder=t_embedder,
        image_projector=image_projector,
        scheduler=scheduler,
        patient_dict=train_patient_dict_all,
        modality_to_latent=modality_to_latent,
        image_latents=image_latents,
        stats=stats, 
        latent_dim=256,
        device=device,
        save_dir=save_dir
    )


if __name__ == "__main__":
    main()

