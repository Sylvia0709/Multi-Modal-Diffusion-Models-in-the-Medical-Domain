import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
import pandas as pd
import numpy as np
import os
import glob # For finding files
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import json # For saving config

# --- SAINT Decoder Model Definition (与参考脚本一致) ---
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
        # self.output_activation = nn.Tanh() # 如果SCNV数据严格在-1到1之间，可以考虑使用

    def forward(self, z):
        x = self.input_proj(z)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaN/Inf detected in model input_proj output (x) during training!")
        for i, (attn_layer, ff_layer) in enumerate(self.layers):
            x_after_attn = attn_layer(x)
            if torch.isnan(x_after_attn).any() or torch.isinf(x_after_attn).any():
                print(f"NaN/Inf detected in model after Attention layer {i} during training!")
            x = ff_layer(x_after_attn)
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"NaN/Inf detected in model after FeedForward layer {i} during training!")
        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        x = self.output_layer(x)
        # if hasattr(self, 'output_activation'):
        #     x = self.output_activation(x)
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("NaN/Inf detected in final model output during training!")
        return x

# --- Custom Dataset ---
class ContinuousTargetDataset(Dataset):
    def __init__(self, latent_tensors, target_data_tensors):
        self.latent_tensors = latent_tensors
        self.target_data_tensors = target_data_tensors
    def __len__(self):
        return len(self.latent_tensors)
    def __getitem__(self, idx):
        return self.latent_tensors[idx], self.target_data_tensors[idx]

# --- Helper function to extract patient ID ---
def extract_patient_id_from_filename(filename):
    base_name = os.path.basename(filename)
    parts = base_name.split('-')
    if len(parts) >= 3:
        id_part3 = parts[2].split('_')[0]
        return f"{parts[0]}-{parts[1]}-{id_part3}"
    return None

# --- Main Training Script for SCNV ---
def main_train_scnv():
    # --- Configuration ---
    latent_dir = "dataset/generated_train_latents"
    # 修改: SCNV 训练数据文件路径
    scnv_file_path = "dataset/scnv_train.csv"
    saved_model_path = "best_decoder_model_scnv.pth"
    config_save_path = "model_config_scnv.json"
    plot_save_path = "training_validation_results_scnv.png"

    decoder_depth = 3; decoder_dim = 128; decoder_heads = 4; decoder_dim_head = 32; decoder_dropout = 0.01
    batch_size = 16; epochs = 200; learning_rate = 1e-4; val_split_ratio = 0.2
    random_seed = 5317; gradient_clipping_value = 1.0; early_stopping_patience = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(random_seed)

    # --- 1. Load Latent Vectors ---
    latent_files = glob.glob(os.path.join(latent_dir, "*.pt"))
    if not latent_files: print(f"Error: No .pt files found in {latent_dir}"); return
    loaded_latents = {}
    print(f"Found {len(latent_files)} .pt files. Attempting to load...")
    for f_path in latent_files:
        patient_id = extract_patient_id_from_filename(f_path)
        if patient_id:
            try:
                tensor = torch.load(f_path, map_location=torch.device('cpu'), weights_only=True)
                if tensor.ndim == 2 and tensor.shape[0] == 1: tensor = tensor.squeeze(0)
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    continue
                loaded_latents[patient_id] = tensor
            except Exception as e: print(f"Warning: Could not load/process latent {f_path}: {e}")
    if not loaded_latents: print("Error: No latent vectors successfully loaded."); return
    print(f"Successfully loaded {len(loaded_latents)} latent vectors.")
    latent_dim = next(iter(loaded_latents.values())).shape[0]
    print(f"Inferred latent_dim: {latent_dim}")

    # --- 2. Load SCNV Data ---
    try:
        # 修改: 读取 SCNV 数据
        scnv_df_original = pd.read_csv(scnv_file_path)
    except FileNotFoundError: print(f"Error: SCNV data file not found: {scnv_file_path}"); return

    if 'patient_id' not in scnv_df_original.columns:
        first_col_name = scnv_df_original.columns[0]
        print(f"Warning: 'patient_id' column not found in {scnv_file_path}. Using first column '{first_col_name}'.")
        scnv_df_original.rename(columns={first_col_name: 'patient_id'}, inplace=True)
    print(f"Loaded {scnv_file_path}.")

    potential_feature_columns = [col for col in scnv_df_original.columns if col != 'patient_id']
    # 修改: SCNV 特征名
    scnv_feature_names = []
    for col in potential_feature_columns:
        try:
            numeric_col_series = pd.to_numeric(scnv_df_original[col], errors='coerce')
            if not numeric_col_series.isnull().all():
                 scnv_feature_names.append(col)
        except Exception:
            print(f"Warning: Could not process column {col} as numeric. Skipping.")
            pass
    if not scnv_feature_names: print("Error: No valid numerical feature columns found for SCNV data."); return
    print(f"Identified {len(scnv_feature_names)} SCNV features.")

    columns_to_keep = ['patient_id'] + scnv_feature_names
    # 修改: SCNV DataFrame
    scnv_df = scnv_df_original[columns_to_keep].copy()
    for col in scnv_feature_names: scnv_df[col] = pd.to_numeric(scnv_df[col], errors='coerce')

    # 训练集NaN用0填充
    if scnv_df[scnv_feature_names].isnull().values.any():
        print("Warning: NaNs in training SCNV features. Filling with 0.")
        scnv_df[scnv_feature_names] = scnv_df[scnv_feature_names].fillna(0)
    if np.isinf(scnv_df[scnv_feature_names].values).any():
        print("Warning: Inf values in training SCNV features. Replacing with 0.")
        scnv_df[scnv_feature_names] = scnv_df[scnv_feature_names].replace([np.inf, -np.inf], 0)

    scnv_df = scnv_df.set_index('patient_id')
    output_dim = len(scnv_feature_names)
    print(f"Output_dim (SCNV features): {output_dim}")

    # --- 3. Align Data and Convert to Tensors ---
    aligned_latents, aligned_scnv_data = [], [] # 修改
    print("Aligning data...")
    for pid, lt in loaded_latents.items():
        if pid in scnv_df.index:
            if lt.shape[0] != latent_dim: continue
            try:
                # 修改: SCNV 值
                scnv_vals = scnv_df.loc[pid, scnv_feature_names].values.astype(np.float32)
                if np.isnan(scnv_vals).any() or np.isinf(scnv_vals).any(): continue
                aligned_latents.append(lt)
                aligned_scnv_data.append(torch.tensor(scnv_vals, dtype=torch.float32)) # 修改
            except Exception as e:
                print(f"Warning: Error converting SCNV data for patient {pid}: {e}") # 修改
                continue
    if not aligned_latents: print("Error: No common patient IDs or data conversion failed."); return
    print(f"Aligned {len(aligned_latents)} samples.")

    all_latents_tensor = torch.stack(aligned_latents)
    all_scnv_tensor = torch.stack(aligned_scnv_data) # 修改

    if torch.isnan(all_latents_tensor).any() or torch.isinf(all_latents_tensor).any(): print("CRITICAL: NaN/Inf in all_latents_tensor!"); return
    if torch.isnan(all_scnv_tensor).any() or torch.isinf(all_scnv_tensor).any(): print("CRITICAL: NaN/Inf in all_scnv_tensor!"); return # 修改

    # --- 4. Split Data (Train/Validation) ---
    # 修改: SCNV 张量
    latents_train, latents_val, scnv_train, scnv_val = train_test_split(
        all_latents_tensor, all_scnv_tensor, test_size=val_split_ratio, random_state=random_seed)
    if len(latents_train) == 0 or len(latents_val) == 0: print("Error: Zero samples after split."); return
    print(f"Data split: {len(latents_train)} train, {len(latents_val)} validation.")

    train_dataset = ContinuousTargetDataset(latents_train, scnv_train) # 修改
    val_dataset = ContinuousTargetDataset(latents_val, scnv_val)     # 修改

    eff_train_bs = min(batch_size, len(train_dataset)) if len(train_dataset) > 0 else 1
    eff_val_bs = min(batch_size, len(val_dataset)) if len(val_dataset) > 0 else 1
    drop_lt = len(train_dataset) > eff_train_bs and len(train_dataset) % eff_train_bs != 0
    train_loader = DataLoader(train_dataset, batch_size=eff_train_bs, shuffle=True, drop_last=drop_lt) if len(train_dataset) > 0 else []
    val_loader = DataLoader(val_dataset, batch_size=eff_val_bs, shuffle=False) if len(val_dataset) > 0 else []

    # --- Model, Loss, Optimizer ---
    model = SAINTDecoder(latent_dim, output_dim, decoder_depth, decoder_dim, decoder_heads, decoder_dim_head, decoder_dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 保存模型配置 ---
    model_config = {
        'latent_dim': latent_dim, 'output_dim': output_dim, 'decoder_depth': decoder_depth,
        'decoder_dim': decoder_dim, 'decoder_heads': decoder_heads, 'decoder_dim_head': decoder_dim_head,
        'decoder_dropout': decoder_dropout, 
        'scnv_feature_names': scnv_feature_names # 修改: 保存 SCNV 特征名
    }
    try:
        with open(config_save_path, "w") as f: json.dump(model_config, f, indent=4)
        print(f"Saved model training configuration to {config_save_path}")
    except Exception as e: print(f"Error saving model configuration: {e}")

    # --- Training Loop ---
    metrics_history = {'train_loss': [], 'val_loss': [],
                       'val_mse_macro': [], 'val_mae_macro': [], 'val_r2_macro': []}
    best_val_metric = float('inf')
    epochs_no_improve = 0
    print(f"\nStarting training for up to {epochs} epochs...")

    for epoch in range(epochs):
        model.train(); running_train_loss = 0.0
        if not train_loader:
            if epoch == 0: print("Warning: Train loader empty.")
        else:
            # 修改: btch_scnv
            for btch_lats, btch_scnv in train_loader:
                btch_lats, btch_scnv = btch_lats.to(device), btch_scnv.to(device) # 修改
                if torch.isnan(btch_lats).any() or torch.isinf(btch_lats).any() or \
                   torch.isnan(btch_scnv).any() or torch.isinf(btch_scnv).any(): # 修改
                    print("Skipping training batch due to NaN/Inf in input/target.")
                    continue
                optimizer.zero_grad()
                predictions = model(btch_lats)
                loss = torch.tensor(float('nan'), device=device)
                if not (torch.isnan(predictions).any() or torch.isinf(predictions).any()):
                    loss = criterion(predictions, btch_scnv) # 修改
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_value)
                    optimizer.step()
                    running_train_loss += loss.item()
        avg_train_loss = running_train_loss / len(train_loader) if train_loader and len(train_loader) > 0 and not np.isnan(running_train_loss) else float('nan')
        metrics_history['train_loss'].append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        all_predictions_list, all_targets_list = [], []
        run_val_loss_epoch = 0.0; valid_val_batches = 0
        if not val_loader:
            if epoch == 0: print("Warning: Validation loader empty.")
        else:
            # 修改: btch_scnv
            for btch_lats, btch_scnv in val_loader:
                btch_lats, btch_scnv = btch_lats.to(device), btch_scnv.to(device) # 修改
                if torch.isnan(btch_lats).any() or torch.isinf(btch_lats).any() or \
                   torch.isnan(btch_scnv).any() or torch.isinf(btch_scnv).any(): # 修改
                    print("Skipping validation batch due to NaN/Inf in input/target.")
                    continue
                predictions = model(btch_lats)
                if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                    print("Skipping validation batch due to NaN/Inf in predictions.")
                    continue
                val_btch_loss = criterion(predictions, btch_scnv) # 修改
                if not torch.isnan(val_btch_loss):
                    run_val_loss_epoch += val_btch_loss.item(); valid_val_batches +=1
                all_predictions_list.append(predictions.cpu())
                all_targets_list.append(btch_scnv.cpu()) # 修改

        avg_val_loss_ep = run_val_loss_epoch / valid_val_batches if valid_val_batches > 0 else float('nan')
        metrics_history['val_loss'].append(avg_val_loss_ep)
        
        current_mse_macro, current_mae_macro, current_r2_macro = float('nan'), float('nan'), float('nan')
        if all_targets_list and all_predictions_list:
            targets_np = torch.cat(all_targets_list, dim=0).detach().numpy()
            preds_np = torch.cat(all_predictions_list, dim=0).detach().numpy()
            if targets_np.size > 0 and preds_np.size > 0 and targets_np.shape == preds_np.shape:
                mse_scores, mae_scores, r2_scores = [], [], []
                skipped_r2 = 0
                for i in range(output_dim): # output_dim is len(scnv_feature_names)
                    target_col, pred_col = targets_np[:, i], preds_np[:, i]
                    mse_scores.append(mean_squared_error(target_col, pred_col))
                    mae_scores.append(mean_absolute_error(target_col, pred_col))
                    if np.var(target_col) > 1e-6 :
                        try: r2_scores.append(r2_score(target_col, pred_col))
                        except ValueError: skipped_r2 +=1
                    else: skipped_r2 +=1
                if mse_scores: current_mse_macro = np.mean(mse_scores)
                if mae_scores: current_mae_macro = np.mean(mae_scores)
                if r2_scores: current_r2_macro = np.mean(r2_scores)
                if (epoch % 20 == 0 or epoch == epochs -1) and skipped_r2 > 0:
                    print(f"Epoch {epoch+1} Skipped features for Val R2: {skipped_r2}/{output_dim}")
            else: print(f"Epoch {epoch+1}: Val targets/preds empty or mismatched shapes.")

        metrics_history['val_mse_macro'].append(current_mse_macro)
        metrics_history['val_mae_macro'].append(current_mae_macro)
        metrics_history['val_r2_macro'].append(current_r2_macro)

        print(f"Epoch [{epoch+1}/{epochs}], TrL: {avg_train_loss:.4f}, VaL: {avg_val_loss_ep:.4f}, "
              f"VaMSE: {current_mse_macro:.4f}, VaMAE: {current_mae_macro:.4f}, VaR2: {current_r2_macro:.4f}")

        if not np.isnan(current_mse_macro):
            if current_mse_macro < best_val_metric:
                best_val_metric = current_mse_macro
                epochs_no_improve = 0
                try:
                    torch.save(model.state_dict(), saved_model_path)
                    print(f"Validation MSE improved. Saved model to {saved_model_path}")
                except Exception as e: print(f"Error saving model: {e}")
            else: epochs_no_improve += 1
        else: epochs_no_improve +=1
        
        if epoch > 10 and all(np.isnan(metrics_history['train_loss'][-3:])): print("Stop: Train loss NaN for 3 epochs."); break
        if epochs_no_improve >= early_stopping_patience: print(f"Early stop: Val MSE no improvement for {early_stopping_patience} epochs."); break
    print("Training finished.")

    # --- Plotting Results ---
    if metrics_history['train_loss']:
        num_plots = sum(1 for v_list in metrics_history.values() if v_list and not all(np.isnan(m) for m in v_list if isinstance(m, float)))
        if num_plots == 0: print("No valid metrics to plot."); return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        plot_idx = 0
        
        titles = ['Training Loss', 'Validation Loss (MSE)', 'Validation MSE (Macro Avg)', 
                  'Validation MAE (Macro Avg)', 'Validation R2 Score (Macro Avg)']
        data_keys = ['train_loss', 'val_loss', 'val_mse_macro', 
                     'val_mae_macro', 'val_r2_macro']
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']

        for i, key in enumerate(data_keys):
            valid_epochs = [ep for ep, m in enumerate(metrics_history[key]) if not np.isnan(m)]
            valid_data = [m for m in metrics_history[key] if not np.isnan(m)]
            if valid_data:
                ax = axes[plot_idx]
                ax.plot(valid_epochs, valid_data, label=titles[i].split('(')[0].strip(), color=colors[i])
                ax.set_xlabel('Epoch'); ax.set_ylabel(titles[i].split('(')[0].strip().split()[-1] if 'Loss' not in titles[i] else 'Loss')
                ax.set_title(titles[i]); ax.legend(); ax.grid(True)
                plot_idx += 1
        
        for i in range(plot_idx, len(axes)): fig.delaxes(axes[i])
        plt.tight_layout()
        plt.savefig(plot_save_path) # 修改: SCNV 图像文件名
        print(f"Saved plot to {plot_save_path}")
    else: print("No data to plot.")

if __name__ == '__main__':
    os.makedirs("dataset/generated_train_latents", exist_ok=True)
    
    # 生成虚拟潜向量文件 (如果不存在)
    if not glob.glob("dataset/generated_train_latents/*.pt"):
        print("Creating dummy latent files for training...")
        for i in range(30):
            patient_id = f"TCGA-TRAIN-S{i:03d}" # 修改ID格式以区分
            dummy_latent = torch.randn(1, 128) # 假设 latent_dim = 128
            torch.save(dummy_latent, f"dataset/generated_train_latents/{patient_id}_latent.pt")

    # 生成虚拟SCNV训练数据 (如果不存在)
    # 修改: 生成 SCNV 虚拟数据
    if not os.path.exists("dataset/scnv_train.csv"):
        print("Creating dummy scnv_train.csv file...")
        num_samples = 35
        num_features = 20 # SCNV 特征数量可能不同
        patient_ids = [f"TCGA-TRAIN-S{i:03d}" for i in range(num_samples - 5)] + \
                      [f"TCGA-OTHER-C{j:03d}" for j in range(5)] # 修改ID格式
        
        data = {'patient_id': patient_ids}
        for j in range(num_features):
            # SCNV 数据通常是离散的整数 (如 0, 1, 2, 3, 4 代表拷贝数状态)
            # 或标准化的连续值。这里用离散值举例，或用连续值。
            # feature_values = np.random.randint(0, 5, size=num_samples) # 例如 0-4 的拷贝数状态
            feature_values = (np.random.rand(num_samples) - 0.5) * 2 # 模拟一些连续值，范围 -1 到 1
            data[f'gene_locus_{j+1}'] = np.clip(feature_values, -2.0, 2.0) # SCNV值范围可能更广
        
        dummy_df = pd.DataFrame(data)
        dummy_df.to_csv("dataset/scnv_train.csv", index=False)

    if not os.path.exists("dataset/scnv_train.csv") or not os.path.exists("dataset/generated_train_latents"):
        print("ERROR: Training data not found. Please ensure scnv_train.csv and latent files exist.")
    else:
        print("Training data found. Proceeding with training for SCNV decoder.") # 修改
        main_train_scnv()