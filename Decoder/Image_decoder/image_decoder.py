import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class MedicalUNetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder shortcut for skip connection (from low-res to high-res)
        self.initial = nn.Conv3d(256, 256, kernel_size=3, padding=1)

        self.up1 = UNetUpBlock(256, 128, scale_factor=(2, 2, 2))       # 12→24, 8→16
        self.up2 = UNetUpBlock(128, 64, scale_factor=(2, 2, 2))        # 24→48, 16→32
        self.up3 = UNetUpBlock(64, 32, scale_factor=(2, 2, 2))         # 48→96, 32→64
        self.up4 = UNetUpBlock(32, 16, scale_factor=(5.33, 3.5, 3.5))  # 96→512, 64→224

        self.final = nn.Conv3d(16, 1, kernel_size=1)
        self.activation = nn.Tanh()

    def forward(self, x):  # x: [B, 768, 256]
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, 12, 8, 8)  # [B, 256, 12, 8, 8]

        x0 = self.initial(x)   # [B, 256, 12, 8, 8]
        x1 = self.up1(x0)      # → [B, 128, 24, 16, 16]
        x2 = self.up2(x1)      # → [B, 64, 48, 32, 32]
        x3 = self.up3(x2)      # → [B, 32, 96, 64, 64]
        x4 = self.up4(x3)      # → [B, 16, 512, 224, 224]

        out = self.final(x4)   # → [B, 1, 512, 224, 224]
        return self.activation(out)

def reconstruction_loss(pred, target, mask, alpha=0.8, eps=1e-8):

    # 1. MSE
    mse = ((pred - target) ** 2 * mask).sum() / (mask.sum() + eps)

    # 2. SSIM
    try:
        ssim_val = ssim(pred * mask, target * mask, data_range=2.0, size_average=True)
    except Exception as e:
        print("SSIM error:", e)
        ssim_val = torch.tensor(0.0, device=pred.device)

    # 3. PSNR
    try:
        psnr = 10 * torch.log10((4.0 + eps) / (mse + eps))  # data_range² = 4
    except Exception:
        psnr = torch.tensor(0.0, device=pred.device)

    # 4. loss
    loss = alpha * mse + (1 - alpha) * (1 - ssim_val)

    return loss, mse.item(), ssim_val.item(), psnr.item()
