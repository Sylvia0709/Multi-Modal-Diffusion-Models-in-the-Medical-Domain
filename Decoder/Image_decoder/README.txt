Medical Image Decoder Training

This module trains a UNet-based medical image decoder for decoding tasks on CT and MR images across various cancer types.

Directory Structure

image_decoder/
│
├── data_loader.py            # Dataset loading code
├── image_decoder.py          # UNet decoder and reconstruction loss definitions
├── infer_and_visualize.py    # Inference and visualization functions
├── train.py                  # Main training script
├── README.md                 # This file
└── requirements.txt          # Required Python packages

Environment Setup

Recommended environment: Google Colab with Python 3.8+.

Install dependencies via:

pip install torch torchvision numpy pandas tqdm pytorch-msssim matplotlib

(When running on Colab, most dependencies are pre-installed.)

Data Preparation

1. Data is stored on Google Drive under:  
   - proj72_data/decoder/organized_train_latents — latent feature directory  
   - proj72_data/xxxx_xx_npz and similar directories containing .npz files

2. Mount Google Drive (in Colab):

from google.colab import drive
drive.mount('/content/drive')

Usage

Run the training script:

python train.py

During training, the following files will be saved:

- Training metrics CSV file, e.g.:  
  /content/drive/MyDrive/proj72_data/decoder/Evaluation_Result/{task}_training_metrics.csv

- Best model weights, e.g.:  
  /content/drive/MyDrive/proj72_data/decoder/Best_model/{task}_medical_unet_decoder.pth

- Visualization results, e.g.:
  /content/drive/MyDrive/proj72_data/decoder/Visualized_images/{task}

Script Overview

- train.py: Main training routine with configurations for multiple cancer types and modalities.  
- data_loader.py: Custom Dataset class loading latent features and target images.  
- image_decoder.py: Defines the UNet decoder architecture and reconstruction loss function.  
- infer_and_visualize.py: Performs model inference and visualization of outputs.

Notes

- Ensure to run Google Drive mounting code (drive.mount) when using Colab.  
- Mount Drive before executing training to avoid path errors.  
- If running locally, adjust paths accordingly and remove Colab-specific code.

Contact

For questions or support, please contact the author.
