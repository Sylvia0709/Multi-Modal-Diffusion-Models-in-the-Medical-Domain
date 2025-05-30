import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import pandas as pd
import numpy as np
import os
import glob
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import json

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=16, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None):
        h = self.heads
        qkv_input = x
        q, k, v = self.to_qkv(qkv_input).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class SAINTDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, depth=4, dim=128, heads=4, dim_head=32, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, dim)
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                PreNorm(dim, Residual(FeedForward(dim, dropout=dropout)))
            ]) for _ in range(depth)
        ])
        self.output_layer = nn.Linear(dim, output_dim)

    def forward(self, z):
        x = self.input_proj(z)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        for i, (attn_layer, ff_layer) in enumerate(self.layers):
            x = ff_layer(attn_layer(x))
        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        x = self.output_layer(x)
        return x

class ContinuousTargetDataset(Dataset):
    def __init__(self, latent_tensors, target_data_tensors):
        self.latent_tensors = latent_tensors
        self.target_data_tensors = target_data_tensors
    def __len__(self):
        return len(self.latent_tensors)
    def __getitem__(self, idx):
        return self.latent_tensors[idx], self.target_data_tensors[idx]

def extract_patient_id_from_filename(filename):
    base_name = os.path.basename(filename)
    name_part, _ = os.path.splitext(base_name)

    id_candidate_from_underscore_split = name_part.split('_')[0]
    parts_from_underscore_split = id_candidate_from_underscore_split.split('-')
    if len(parts_from_underscore_split) >= 3:
        return f"{parts_from_underscore_split[0]}-{parts_from_underscore_split[1]}-{parts_from_underscore_split[2]}"

    parts_direct_split = name_part.split('-')
    if len(parts_direct_split) >= 3:
        return f"{parts_direct_split[0]}-{parts_direct_split[1]}-{parts_direct_split[2]}"
    
    return None

def main_inference_scnv():
    test_latent_dir = "dataset/test_scnv_latents"
    scnv_test_file_path = "dataset/scnv_test.csv"
    model_load_path = "best_decoder_model_scnv.pth"
    config_load_path = "model_config_scnv.json"
    
    output_folder = "result"
    os.makedirs(output_folder, exist_ok=True)
    results_plot_path = os.path.join(output_folder, "inference_results_scnv.png")
    reconstructed_scnv_output_path = os.path.join(output_folder, "reconstructed_scnv_predictions.csv")
    
    batch_size = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(config_load_path):
        print(f"CRITICAL Error: Model configuration file not found at {config_load_path}")
        return
    
    with open(config_load_path, "r") as f:
        model_config = json.load(f)

    latent_dim = model_config['latent_dim']
    output_dim = model_config['output_dim']
    decoder_depth = model_config['decoder_depth']
    decoder_dim = model_config['decoder_dim']
    decoder_heads = model_config['decoder_heads']
    decoder_dim_head = model_config['decoder_dim_head']
    decoder_dropout = model_config['decoder_dropout']

    scnv_feature_names_from_train = model_config.get('scnv_feature_names')
    if scnv_feature_names_from_train is None:
        scnv_feature_names_from_train = model_config.get('methylation_feature_names')
    if scnv_feature_names_from_train is None:
        print("CRITICAL Error: SCNV feature names (e.g., 'scnv_feature_names') not found in config file.")
        return

    latent_files_test = glob.glob(os.path.join(test_latent_dir, "*.pt"))
    if not latent_files_test:
        print(f"CRITICAL Error: No .pt files found in {test_latent_dir}")
        return

    loaded_latents_test = {}
    for f_path in latent_files_test:
        patient_id = extract_patient_id_from_filename(f_path)
        if patient_id:
            tensor = torch.load(f_path, map_location=torch.device('cpu'), weights_only=True)
            if tensor.ndim == 2 and tensor.shape[0] == 1: tensor = tensor.squeeze(0)
            if tensor.ndim == 0:
                print(f"Warning: Loaded scalar tensor from {f_path}. Skipping.")
                continue
            if tensor.shape[0] != latent_dim:
                print(f"Warning: Latent file {f_path} has dimension {tensor.shape[0]}, model expects {latent_dim}. Skipping.")
                continue
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"Warning: NaN/Inf values found in loaded test latent file: {f_path}. Skipping.")
                continue
            loaded_latents_test[patient_id] = tensor
        else:
            print(f"Warning: Could not extract patient ID from filename {f_path}. Skipping.")


    if not loaded_latents_test:
        print("CRITICAL Error: No test latent vectors successfully loaded or matched expected dimension.")
        return

    scnv_df_test_original = pd.read_csv(scnv_test_file_path)

    if 'patient_id' not in scnv_df_test_original.columns:
        if scnv_df_test_original.columns.empty:
            print(f"CRITICAL Error: Test SCNV file {scnv_test_file_path} is empty or has no columns.")
            return
        first_col_name = scnv_df_test_original.columns[0]
        print(f"Warning: 'patient_id' column not found in {scnv_test_file_path}. Using first column '{first_col_name}' as patient_id.")
        scnv_df_test_original.rename(columns={first_col_name: 'patient_id'}, inplace=True)

    missing_cols = [col for col in scnv_feature_names_from_train if col not in scnv_df_test_original.columns]
    if missing_cols:
        print(f"CRITICAL Error: The following SCNV features (required by model) are missing in test file {scnv_test_file_path}: {missing_cols}")
        return

    columns_to_keep_test = ['patient_id'] + scnv_feature_names_from_train
    scnv_df_test = scnv_df_test_original[columns_to_keep_test].copy()

    for col in scnv_feature_names_from_train:
        scnv_df_test[col] = pd.to_numeric(scnv_df_test[col], errors='coerce')

    if scnv_df_test[scnv_feature_names_from_train].isnull().values.any():
        print("Warning: NaNs found in test SCNV features after numeric conversion. Filling with 0.")
        scnv_df_test[scnv_feature_names_from_train] = scnv_df_test[scnv_feature_names_from_train].fillna(0)

    if np.isinf(scnv_df_test[scnv_feature_names_from_train].values).any():
        print("Warning: Inf values found in test SCNV features. Replacing with 0.")
        scnv_df_test[scnv_feature_names_from_train] = scnv_df_test[scnv_feature_names_from_train].replace([np.inf, -np.inf], 0)

    scnv_df_test = scnv_df_test.set_index('patient_id')

    if len(scnv_feature_names_from_train) != output_dim:
        print(f"CRITICAL Error: Mismatch in output dimension. Config expects {output_dim}, found {len(scnv_feature_names_from_train)} from training feature names.")
        return

    aligned_latents_test_list, aligned_scnv_test_list = [], []
    valid_pids_for_test = []

    for pid, latent_tensor in loaded_latents_test.items():
        if pid in scnv_df_test.index:
            scnv_values_series = scnv_df_test.loc[pid, scnv_feature_names_from_train]
            scnv_values_np = scnv_values_series.values.astype(np.float32)

            if np.isnan(scnv_values_np).any() or np.isinf(scnv_values_np).any():
                print(f"Warning: NaN/Inf values found in processed SCNV data for test patient {pid} AFTER imputation. Skipping.")
                continue

            aligned_latents_test_list.append(latent_tensor)
            aligned_scnv_test_list.append(torch.tensor(scnv_values_np, dtype=torch.float32))
            valid_pids_for_test.append(pid)
        else:
            print(f"Warning: Patient ID {pid} from latent files not found in SCNV data. Skipping.")


    if not aligned_latents_test_list:
        print("CRITICAL Error: No common patient IDs found or all samples had issues.")
        return

    all_latents_test_tensor = torch.stack(aligned_latents_test_list)
    all_scnv_test_tensor = torch.stack(aligned_scnv_test_list)

    if torch.isnan(all_latents_test_tensor).any() or torch.isinf(all_latents_test_tensor).any():
        print("CRITICAL ERROR: NaN/Inf values detected in 'all_latents_test_tensor'!")
        return
    if torch.isnan(all_scnv_test_tensor).any() or torch.isinf(all_scnv_test_tensor).any():
        print("CRITICAL ERROR: NaN/Inf values detected in 'all_scnv_test_tensor'!")
        return

    test_dataset_inference = ContinuousTargetDataset(all_latents_test_tensor, all_scnv_test_tensor)
    if len(test_dataset_inference) == 0:
        print("CRITICAL Error: Test dataset is empty after alignment and processing.")
        return
    eff_test_bs = min(batch_size, len(test_dataset_inference))
    test_loader_inference = DataLoader(test_dataset_inference, batch_size=eff_test_bs, shuffle=False)

    model = SAINTDecoder(
        latent_dim=latent_dim, output_dim=output_dim, depth=decoder_depth,
        dim=decoder_dim, heads=decoder_heads, dim_head=decoder_dim_head, dropout=decoder_dropout
    ).to(device)

    if not os.path.exists(model_load_path):
        print(f"CRITICAL Error: Saved model file not found at {model_load_path}")
        return
    
    model.load_state_dict(torch.load(model_load_path, map_location=device))
    model.eval()

    all_predictions_list_inf, all_targets_list_inf = [], []

    with torch.no_grad():
        for batch_idx, (btch_lats, btch_scnv_targets) in enumerate(test_loader_inference):
            btch_lats = btch_lats.to(device)
            if torch.isnan(btch_lats).any() or torch.isinf(btch_lats).any():
                print(f"Warning: NaN/Inf in batch latents (batch {batch_idx}). Skipping.")
                continue
            out_predictions = model(btch_lats)
            if torch.isnan(out_predictions).any() or torch.isinf(out_predictions).any():
                print(f"Warning: NaN/Inf in model predictions (batch {batch_idx}). Skipping.")
                continue
            all_predictions_list_inf.append(out_predictions.cpu())
            all_targets_list_inf.append(btch_scnv_targets.cpu())

    if not all_predictions_list_inf or not all_targets_list_inf:
        print("CRITICAL Error: No valid predictions or targets collected.")
        return
    if len(all_predictions_list_inf) != len(all_targets_list_inf):
        print("CRITICAL Error: Mismatch in collected prediction and target batches.")
        return

    predictions_np_inf = torch.cat(all_predictions_list_inf, dim=0).numpy()
    targets_np_inf = torch.cat(all_targets_list_inf, dim=0).numpy()

    if targets_np_inf.size == 0 or predictions_np_inf.size == 0:
        print("CRITICAL Error: Targets or predictions arrays are empty.")
        return
    if targets_np_inf.shape != predictions_np_inf.shape:
        print(f"CRITICAL Error: Shape mismatch: targets {targets_np_inf.shape}, predictions {predictions_np_inf.shape}.")
        return

    if len(valid_pids_for_test) == predictions_np_inf.shape[0] and len(scnv_feature_names_from_train) == predictions_np_inf.shape[1]:
        reconstructed_scnv_df = pd.DataFrame(predictions_np_inf, index=valid_pids_for_test, columns=scnv_feature_names_from_train)
        reconstructed_scnv_df.index.name = 'patient_id'
        reconstructed_scnv_df.to_csv(reconstructed_scnv_output_path)
        print(f"Saved reconstructed SCNV predictions to {reconstructed_scnv_output_path}")
    else:
        print("CRITICAL Error: Mismatch in dimensions for saving reconstructed SCNV table.")
        print(f"  Patient IDs: {len(valid_pids_for_test)}, Predictions rows: {predictions_np_inf.shape[0]}")
        print(f"  Feature names: {len(scnv_feature_names_from_train)}, Predictions columns: {predictions_np_inf.shape[1]}")


    mse_inf_flat = mean_squared_error(targets_np_inf.flatten(), predictions_np_inf.flatten())
    rmse_inf_flat = np.sqrt(mse_inf_flat)
    mae_inf_flat = mean_absolute_error(targets_np_inf.flatten(), predictions_np_inf.flatten())
    r2_scores_per_feature_inf = []
    skipped_features_r2_inf = 0
    if output_dim > 0 :
        for i in range(output_dim):
            target_col, pred_col = targets_np_inf[:, i], predictions_np_inf[:, i]
            if np.var(target_col) > 1e-6:
                r2_scores_per_feature_inf.append(r2_score(target_col, pred_col))
            else:
                skipped_features_r2_inf += 1
        r2_macro_inf = np.mean(r2_scores_per_feature_inf) if r2_scores_per_feature_inf else float('nan')
    else:
        r2_macro_inf = float('nan')

    print("\n--- SCNV Inference Results ---")
    print(f"Test RMSE (Overall Flattened): {rmse_inf_flat:.4f}")
    print(f"Test MSE (Overall Flattened): {mse_inf_flat:.4f}")
    print(f"Test MAE (Overall Flattened): {mae_inf_flat:.4f}")
    print(f"Test R2 (Macro Average over Features): {r2_macro_inf:.4f}")
    if output_dim > 0 and skipped_features_r2_inf > 0:
        print(f"Skipped features for R2 calculation (low/zero variance): {skipped_features_r2_inf}/{output_dim}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax1 = axes[0]
    sample_size_plot = min(len(targets_np_inf.flatten()), 5000)
    if sample_size_plot > 0:
        indices_plot = np.random.choice(len(targets_np_inf.flatten()), sample_size_plot, replace=False)
        ax1.scatter(targets_np_inf.flatten()[indices_plot], predictions_np_inf.flatten()[indices_plot], alpha=0.3, s=10, label="Samples")
    else:
        ax1.text(0.5, 0.5, "No data points to plot", ha='center', va='center', transform=ax1.transAxes)

    min_val = 0
    max_val = 1
    if targets_np_inf.size > 0 and predictions_np_inf.size > 0:
        min_val = min(targets_np_inf.min(), predictions_np_inf.min())
        max_val = max(targets_np_inf.max(), predictions_np_inf.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal y=x')
    ax1.set_xlabel("Actual SCNV Values (Flattened, Sampled)")
    ax1.set_ylabel("Predicted SCNV Values (Flattened, Sampled)")
    ax1.set_title(f"Actual vs. Predicted SCNV (Test Set)\nRMSE: {rmse_inf_flat:.4f}")
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')

    ax2 = axes[1]
    if targets_np_inf.size > 0:
        residuals = (predictions_np_inf - targets_np_inf).flatten()
        ax2.hist(residuals, bins=60, edgecolor='black', alpha=0.75, density=True)
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        ax2.text(0.05, 0.95, f"Mean: {mean_residual:.3f}\nStd: {std_residual:.3f}",
                 transform=ax2.transAxes, va='top', ha='left', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
    else:
        ax2.text(0.5, 0.5, "No residuals to plot", ha='center', va='center', transform=ax2.transAxes)
    ax2.set_xlabel("Prediction Error (Predicted - Actual)")
    ax2.set_ylabel("Density")
    ax2.set_title("Histogram of Prediction Residuals (Test Set)")
    ax2.grid(True)
    ax2.axvline(0, color='r', linestyle='--', lw=2, label='Zero Error')
    ax2.legend()

    plt.suptitle("SAINT Decoder Inference Performance on SCNV Data", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(results_plot_path, dpi=150)
    print(f"Saved inference results plot to {results_plot_path}")

if __name__ == '__main__':
    test_latent_dir_check = "dataset/final_generated_test_latents"
    scnv_test_file_check = "dataset/scnv_test.csv"
    model_file_check = "best_decoder_model_scnv.pth"
    config_file_check = "model_config_scnv.json"

    required_dirs_inf = ["dataset", test_latent_dir_check, "result"] # Ensure result directory is checked/created
    for req_dir in required_dirs_inf:
        if not os.path.exists(req_dir):
            os.makedirs(req_dir, exist_ok=True)


    missing_files_flag = False
    for f_path, name in [(scnv_test_file_check, "Test SCNV data"),
                         (model_file_check, "Trained SCNV model file"),
                         (config_file_check, "SCNV Model configuration file")]:
        if not os.path.exists(f_path):
            print(f"- CRITICAL: Missing {name}: {f_path}")
            missing_files_flag = True
    if not glob.glob(os.path.join(test_latent_dir_check, "*.pt")):
        print(f"- CRITICAL: No .pt files found in test latent directory: {test_latent_dir_check}")
        missing_files_flag = True

    if missing_files_flag:
        print("ERROR: One or more critical files/data are missing for SCNV inference.")
    else:
        print("Required files found. Proceeding with SCNV inference...")
        main_inference_scnv()