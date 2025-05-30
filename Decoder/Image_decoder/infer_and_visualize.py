import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def save_side_by_side(pred, target, save_path):
    pred_np = pred.squeeze().cpu().numpy()      # [D, H, W]
    target_np = target.squeeze().cpu().numpy()

    mid = pred_np.shape[0] // 2
    pred_slice = pred_np[mid]
    target_slice = target_np[mid]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(target_slice, cmap='gray')
    axes[0].set_title("Ground Truth (mid slice)")
    axes[1].imshow(pred_slice, cmap='gray')
    axes[1].set_title("Prediction (mid slice)")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Middle comparison image saved: {save_path}")

def save_all_slices_side_by_side(pred, target, save_path):
    """
    pred, target: torch.Tensor, shape [D, H, W]
    save_path: str
    """
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    D = pred_np.shape[0]

    fig, axes = plt.subplots(2, D, figsize=(D*2, 4))

    for i in range(D):
        axes[0, i].imshow(target_np[i], cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title("Ground Truth")

        axes[1, i].imshow(pred_np[i], cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title("Prediction")
    filename = os.path.basename(save_path)         # "compare.png"
    dirname = os.path.dirname(save_path)           # "" or folder path
    new_filename = "all_" + filename               # "all_compare.png"
    new_save_path = os.path.join(dirname, new_filename)


    plt.tight_layout()
    plt.savefig(new_save_path)
    plt.close()
    print(f"All slices comparison saved: {new_save_path}")

def visualize(basename, pred_tensor, target_npz_dir, save_dir, device='cpu'):
    """
    basename: str, e.g., "TCGA-ZF-A9RN"
    pred_tensor: torch.Tensor, shape [1, 1, D, H, W]
    target_npz_dir: str
    save_dir: str
    """
    os.makedirs(save_dir, exist_ok=True)
    patient_id = basename.split('_')[0]

    target_path = os.path.join(target_npz_dir, f"{patient_id}.npz")
    save_path = os.path.join(save_dir, f"{basename}_compare_output.png")

    if not os.path.exists(target_path):
        print(f"skip, target file not found: {target_path}")
        return

    data = np.load(target_path)
    img = data["image"]            # (1, H, W, D)
    D = img.shape[-1]              # real depth

    pred_cropped = pred_tensor[:, :, :D, :, :]
    target = torch.tensor(img).permute(0, 3, 1, 2).to(device)  # [1, D, H, W]

    save_side_by_side(pred_cropped[0, 0], target[0], save_path)
    save_all_slices_side_by_side(pred_cropped[0, 0], target[0], save_path)

def inference(decoder, latent_dir, save_dir, target_npz_dir=None, vis_save_dir=None, device='cpu'):
    os.makedirs(save_dir, exist_ok=True)
    if vis_save_dir is not None:
        os.makedirs(vis_save_dir, exist_ok=True)

    decoder.eval()
    files = [f for f in os.listdir(latent_dir) if f.endswith('.pt')]

    with torch.no_grad():
        for fname in tqdm(files, desc="Inference"):
            path = os.path.join(latent_dir, fname)
            latent = torch.load(path).unsqueeze(0).to(device)
            output = decoder(latent)

            if target_npz_dir is not None and vis_save_dir is not None:
                try:
                    visualize(fname.replace('.pt', ''), output.cpu(), target_npz_dir, vis_save_dir, device)
                except Exception as e:
                    print(f"[!] Visualization failed for {fname}: {e}")
