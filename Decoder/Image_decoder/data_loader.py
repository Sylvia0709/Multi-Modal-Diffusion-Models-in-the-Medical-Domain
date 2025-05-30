import os
import numpy as np
import torch
from torch.utils.data import Dataset


def load_npz_image(path, max_depth=512):
    data = np.load(path)
    img = data["image"]  # shape: (1, 224, 224, D)

    img = 2 * (img - img.min()) / (img.max() - img.min()) - 1  # Normalize to [-1, 1]

    D = img.shape[-1]
    assert img.shape[:3] == (1, 224, 224), f"Shape mismatch: {img.shape}"

    # padding
    padded = np.zeros((1, 224, 224, max_depth), dtype=np.float32)
    padded[..., :D] = img

    # mask
    mask = np.zeros((1, max_depth, 224, 224), dtype=np.float32)
    mask[:, :D, :, :] = 1

    # convert to PyTorch: (1, D, 224, 224)
    padded = torch.tensor(padded).permute(0, 3, 1, 2)   # [1, D, 224, 224]
    mask = torch.tensor(mask)

    return padded, mask

class MedicalDecoderDataset(Dataset):
    def __init__(self, npz_dir, feature_dir, max_depth=512):
        self.npz_dir = npz_dir
        self.feature_dir = feature_dir
        self.max_depth = max_depth

        self.samples = []

        for fname in os.listdir(feature_dir):
            if fname.endswith('.pt'):
                # obtain patient_id，for example：blca123_t1_input.pt -> blca123
                patient_id = fname.split('_')[0]

                npz_path = os.path.join(npz_dir, f"{patient_id}.npz")
                pt_path = os.path.join(feature_dir, fname)

                if os.path.exists(npz_path):
                    self.samples.append((npz_path, pt_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npz_path, pt_path = self.samples[idx]

        image, mask = load_npz_image(npz_path, self.max_depth)  # [1, D, 224, 224]
        latent = torch.load(pt_path)  # [768, 256]

        return latent, image, mask