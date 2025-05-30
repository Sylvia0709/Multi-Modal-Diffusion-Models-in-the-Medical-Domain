import os
import gc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import gdown

from data_loader import MedicalDecoderDataset
from image_decoder import MedicalUNetDecoder, reconstruction_loss
from infer_and_visualize import inference


def train_model(npz_dir, feature_dir, num_epochs, batch_size, device, task='blca_ct'):

    use_amp = True

    dataset = MedicalDecoderDataset(npz_dir=npz_dir, feature_dir=feature_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    decoder = MedicalUNetDecoder().to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    records = {"epoch": [], "loss": [], "mse": [], "ssim": [], "psnr": []}

    for epoch in range(num_epochs):
        decoder.train()
        total_loss = total_mse = total_ssim = total_psnr = 0

        for latent, target, mask in tqdm(loader, desc=f"Epoch {epoch+1}"):
            latent, target, mask = latent.to(device), target.to(device), mask.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                output = decoder(latent)
                if output.shape[2] != target.shape[2]:
                    diff = target.shape[2] - output.shape[2]
                    if diff > 0:
                        output = F.pad(output, (0, 0, 0, 0, 0, diff))
                    else:
                        output = output[:, :, :target.shape[2], :, :]
                loss, mse_val, ssim_val, psnr_val = reconstruction_loss(output, target, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_mse += mse_val
            total_ssim += ssim_val
            total_psnr += psnr_val

            gc.collect()
            torch.cuda.empty_cache()

        N = len(loader)
        records["epoch"].append(epoch + 1)
        records["loss"].append(total_loss / N)
        records["mse"].append(total_mse / N)
        records["ssim"].append(total_ssim / N)
        records["psnr"].append(total_psnr / N)

    df = pd.DataFrame(records)
    csv_save_path = os.path.join("/content/drive/MyDrive/proj72_data/decoder/Evaluation_Result", task+"_training_metrics.csv")
    df.to_csv(csv_save_path, index=False)

    decoder.eval()

    # save
    model_save_path = os.path.join("/content/drive/MyDrive/proj72_data/decoder/Best_model", task+"_medical_unet_decoder.pth")
    torch.save(decoder.state_dict(), model_save_path)

    # infer
    inference(
    decoder,
    latent_dir=feature_dir,
    save_dir=task+"_output",
    target_npz_dir=npz_dir,
    vis_save_dir=os.path.join("/content/drive/MyDrive/proj72_data/decoder/Visualized_images", task),
    device=device
  )



def main():
    from google.colab import drive
    drive.mount('/content/drive')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_base_dir = "/content/drive/MyDrive/proj72_data/decoder/organized_train_latents"
    npz_base_dir = "/content/drive/MyDrive/proj72_data"

    train_model(
        npz_dir=os.path.join(npz_base_dir, "blca_mr_npz"),
        feature_dir=os.path.join(feature_base_dir, "BLCA_MR"),
        num_epochs=50,
        batch_size=2,
        device=device,
        task="blca_mr"
    )

    train_model(
        npz_dir=os.path.join(npz_base_dir, "blca_ct_npz"),
        feature_dir=os.path.join(feature_base_dir, "BLCA_CT"),
        num_epochs=70,
        batch_size=1,
        device=device,
        task="blca_ct"
    )

    train_model(
        npz_dir=os.path.join(npz_base_dir, "kirc_ct_npz"),
        feature_dir=os.path.join(feature_base_dir, "KIRC_CT"),
        num_epochs=50,
        batch_size=1,
        device=device,
        task="kirc_ct"
    )

    train_model(
        npz_dir=os.path.join(npz_base_dir, "kirc_mr_npz"),
        feature_dir=os.path.join(feature_base_dir, "KIRC_MR"),
        num_epochs=50,
        batch_size=1,
        device=device,
        task="kirc_mr"
    )

    train_model(
        npz_dir=os.path.join(npz_base_dir, "lihc_ct_npz"),
        feature_dir=os.path.join(feature_base_dir, "LIHC_CT"),
        num_epochs=50,
        batch_size=1,
        device=device,
        task="lihc_ct"
    )

    train_model(
        npz_dir=os.path.join(npz_base_dir, "lihc_mr_npz"),
        feature_dir=os.path.join(feature_base_dir, "LIHC_MR"),
        num_epochs=50,
        batch_size=2,
        device=device,
        task="lihc_mr"
    )
if __name__ == "__main__":
    main()