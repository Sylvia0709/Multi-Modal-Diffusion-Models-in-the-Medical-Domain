# data_loader.py
import os
import glob
import torch
import numpy as np
import pandas as pd
import ast
from torch.utils.data import Dataset

def load_data(omic_path, image_path, task_path, MODALITY_LIST):
    # --- 1. load latents,tasks and patients ---
    clinical = pd.read_csv(os.path.join(omic_path, "clinical_features.csv"), index_col=0).drop(columns=["cancer_type"])
    mutation = pd.read_csv(os.path.join(omic_path, "mutation_features.csv"), index_col=0).drop(columns=["cancer_type"])
    methylation = pd.read_csv(os.path.join(omic_path, "methylation_features.csv"), index_col=0).drop(columns=["cancer_type"])
    scnv = pd.read_csv(os.path.join(omic_path, "scnv_features.csv"), index_col=0).drop(columns=["cancer_type"])

    multimodal_tasks = pd.read_csv(os.path.join(task_path, "multimodal_tasks.csv"))

    image_modalities = ["ct", "mr"]
    cancer_types = ["blca", "kirc", "lihc"]
    image_latents = {}

    for cancer in cancer_types:
        for modality in image_modalities:
            folder = os.path.join(image_path, "image_features", f"{cancer}_{modality}")
            if not os.path.exists(folder):
                continue

            pt_files = glob.glob(os.path.join(folder, "*.pt"))
            for pt_file in pt_files:
                filename = os.path.basename(pt_file)
                patient_id = filename.replace("_features.pt", "")
                latent = torch.load(pt_file, map_location="cpu", weights_only=False)
                if hasattr(latent, "as_tensor"):
                    latent = latent.as_tensor()
                if patient_id not in image_latents:
                    image_latents[patient_id] = {}
                image_latents[patient_id][modality.upper()] = latent

    stats = {}
    for m, df in [("mutation", mutation), ("methylation", methylation), ("scnv", scnv), ("clinical", clinical)]:
        v = torch.from_numpy(df.values.flatten()).float()
        stats[m] = (v.mean().item(), v.std().item())

    for m in ["CT", "MR"]:
        allv = []
        for imgs in image_latents.values():
            if m in imgs:
                val = imgs[m]
                if isinstance(val, np.ndarray):
                    t = torch.from_numpy(val.flatten()).float()
                else:
                    t = val.flatten().float()
                allv.append(t)
        if allv:
            allv = torch.cat(allv, dim=0)
            stats[m] = (allv.mean().item(), allv.std().item())
        else:
            stats[m] = (0.0, 1.0)

    patient_dict = {}

    for _, row in multimodal_tasks.iterrows():
        cancer = row["cancer_type"]
        input_modalities = eval(row["input"])
        target_modality = row["target"]
        train_pids = eval(row["train_patients"])
        test_pids = eval(row["test_patients"])

        for split, pid_list in [("train", train_pids), ("test", test_pids)]:
            for pid in pid_list:

                if pid not in patient_dict:
                    patient_dict[pid] = {
                        "cancer_type": cancer,
                        "split": split,
                        "tasks": [],              #  task
                        "modalities": []
                    }

                # add task to patient dict
                patient_dict[pid]["tasks"].append({
                    "input_modalities": input_modalities,
                    "target_modality": target_modality
                })

                # add modalities to patient dict
                for m in input_modalities + [target_modality]:
                    if m in MODALITY_LIST and m not in patient_dict[pid]["modalities"]:
                        patient_dict[pid]["modalities"].append(m)

                for m in ["mutation", "methylation", "scnv", "clinical"]:
                    if m in patient_dict[pid]["modalities"]:
                        df = {"mutation": mutation,
                            "methylation": methylation,
                            "scnv": scnv,
                            "clinical": clinical}[m]
                        if pid in df.index and m not in patient_dict[pid]:
                            raw = df.loc[pid].values.astype(np.float32)
                            μ, σ = stats[m]
                            patient_dict[pid][m] = ((raw - μ) / (σ + 1e-8)).astype(np.float32)

                if pid in image_latents:
                    for m in ["CT", "MR"]:
                        if m in patient_dict[pid]["modalities"]:
                            val = image_latents[pid].get(m, None)
                            if val is not None and m not in patient_dict[pid]:
                                if hasattr(val, "cpu"):
                                    val = val.cpu().numpy()
                                raw = val.astype(np.float32)
                                μ, σ = stats[m]
                                patient_dict[pid][m] = ((raw - μ) / (σ + 1e-8)).astype(np.float32)

    return patient_dict, image_latents, stats, multimodal_tasks

class MultiModalTranslationDataset(Dataset):
    def __init__(self, task_table, patient_dict, split="train"):
        self.task_table = task_table
        self.patient_dict = patient_dict
        self.split = split

        self.samples = []

        # process tasks
        for _, row in self.task_table.iterrows():
            cancer_type = row["cancer_type"]
            input_modalities = ast.literal_eval(row["input"])
            target_modality = row["target"]

            if split == "train":
                patients = ast.literal_eval(row["train_patients"])
            elif split == "test":
                patients = ast.literal_eval(row["test_patients"])
            else:
                raise ValueError("split must be train or test")

            for pid in patients:
                if pid not in patient_dict:
                    continue

                patient_data = patient_dict[pid]

                if target_modality not in patient_data:
                    continue

                has_all_inputs = all(m in patient_data for m in input_modalities)
                if not has_all_inputs:
                    continue

                self.samples.append({
                    "patient_id": pid,
                    "cancer_type": cancer_type,
                    "input_modalities": input_modalities,
                    "target_modality": target_modality,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        pid = item["patient_id"]
        patient_data = self.patient_dict[pid]

        # input modalities
        inputs = []
        for m in item["input_modalities"]:
            latent = patient_data[m]
            if isinstance(latent, np.ndarray):
                latent = torch.tensor(latent, dtype=torch.float32)
            inputs.append(latent)

        # target latent
        target = patient_data[item["target_modality"]]
        if isinstance(target, np.ndarray):
            target = torch.tensor(target, dtype=torch.float32)

        return {
            "inputs": inputs,                          # list of tensors
            "target": target,                          # tensor
            "target_modality": item["target_modality"],
            "cancer_type": item["cancer_type"]
        }

def split_data(patient_dict):
    
    # obtain patient id for training set
    train_patients_all = [p for p in patient_dict if patient_dict[p]["split"] == "train"]

    # split into train(90) ans validation(10)
    split_idx = int(len(train_patients_all) * 0.9)
    train_patients = train_patients_all[:split_idx]
    val_patients = train_patients_all[split_idx:]

    # create patient dict for train and val
    train_patient_dict = {p: patient_dict[p] for p in train_patients}
    val_patient_dict = {p: patient_dict[p] for p in val_patients}
    
    return train_patients, val_patients, train_patient_dict, val_patient_dict