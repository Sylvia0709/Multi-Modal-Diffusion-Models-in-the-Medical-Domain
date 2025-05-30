MULTI-MODAL DIFFUSION PIPELINE FOR MEDICAL CANCER DATA COMPLETION
=============================================================================

A reproducible research code-base that **predicts missing clinical or imaging
modalities** and **generates synthetic latent representations** for cancer
patients by coupling
  1. rigorous data pre-processing,
  2. deep feature extraction for tabular and 3-D radiology data, and
  3. a diffusion-based encoder–decoder architecture.


1. QUICK START
--------------

    # 1. clone & create the environment
    git clone https://github.com/your-org/mm-diffusion-pipeline.git
    cd mm-diffusion-pipeline
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt       # or: conda env create -f environment.yml

    # 2. download TCGA sample data (BLCA / KIRC / LIHC)
    bash scripts/download_tcga.sh

    # 3. end-to-end run (pre-processing → feature extraction → training → evaluation)
    bash scripts/run_all.sh               # see scripts/ for individual stage commands


2. DIRECTORY LAYOUT
-------------------

.
├── 01-OriginalDataset/                raw TCGA CT/MR and omics files
├── 02-Data_process_and_dataset/       notebooks & scripts that create cleaned datasets
├── 03-Encoder/                        UNet encoder + cross-modal attention
├── 04-Diffusion_model/                noise scheduler, forward / reverse process
├── 05-Decoder/                        UNet medical image decoder
├── 06-Compare_model/                  baselines (VAE, GAN, TabNet …)
├── 07-Evaluation/                     metrics, visualisation, statistical tests
├── 08-Output/                         checkpoints/, runs/, results/
├── config/                            YAML hyper-parameter files
├── scripts/                           one-click bash / Python helpers
├── tests/                             unit & integration tests (pytest)
└── README.txt                         YOU ARE HERE


3. DATA PRE-PROCESSING PIPELINE
-------------------------------

Stage                    Script / Notebook               Purpose
-----------------------  ------------------------------  ---------------------------------------
Tabular clean-up         preprocess_tab.ipynb            transpose, MinMax-scale and ID-normalise TCGA tables
Image metadata merge     merge_img_metadata.ipynb        join metadata.csv & manifest.xlsx; keep CT/MR only
Series sub-selection     subselect_images.py             copy valid DICOM series, enforce modality checks
DICOM → NPZ              v2_dcm_to_npz.py                largest coherent chunk, resize to 224², save volume


4. FEATURE EXTRACTION
---------------------

Modality     Extractor (location)                 Output
-----------  -----------------------------------  -----------------
Tabular      TabTransformer (tab_transformer.py)  .npy feature arrays
3-D Images   Swin-UNet encoder (swin_feature.py)  .pt tensors


5. MODEL ARCHITECTURE
---------------------

  • Encoder – projects any subset of available modalities into a common latent space  
    using cross-modal attention.  
  • Conditional Diffusion Core – learns the distribution of complete patient latent
    representations and stochastically imputes missing parts.  
  • Decoder – modality-specific UNet that reconstructs realistic CT/MR volumes from
    predicted latents.


6. TRAINING & INFERENCE
-----------------------

    # (A) train diffusion model
    python main.py --config config/diffusion.yaml

    # (B) train medical image decoder
    python 05-Decoder/train.py --config config/decoder.yaml

    # (C) impute missing modalities + generate outputs
    python train_generate.py --mode generate \
                             --checkpoint checkpoints/diffusion_best.pt

Checkpoints:  08-Output/checkpoints/  
TensorBoard:  08-Output/runs/


7. EVALUATION
-------------

Run quantitative evaluation and visual inspection:

    python 07-Evaluation/evaluate.py --data_dir 08-Output/results/

The script reports PSNR / SSIM for reconstructed images and AUROC / F1 for clinical
prediction tasks. Visualised slices are saved to 08-Output/results/vis/.


8. REQUIREMENTS
---------------

Core dependencies (see requirements.txt for full list):

    torch >=1.10, torchvision, numpy, pandas, tqdm, monai,
    diffusers, matplotlib, pytorch-msssim


9. REPRODUCIBILITY CHECKLIST
----------------------------

• All configurable hyper-parameters live in YAML files under config/.  
• Random seeds fixed (torch, numpy, random) for every run.  
• Versioned datasets stored under 02-Data_process_and_dataset/processed/.  
• Unit tests (pytest -q) must pass before CI merges.


10. LICENSE
-----------

The code is released for **academic research only**; see LICENSE for terms.


11. CONTACT
-----------

Questions or pull-requests are welcome—open an issue or email first.last@university.edu.


12. LARGE FILE STORAGE (Google Drive)
------------------------------------

GitHub imposes a 100 MB per-file limit and a 5 GB total quota for free
Git LFS. To keep the repo lightweight, **all assets larger than 100 MB
(checkpoints, pre-processed imaging volumes, etc.) are stored externally
and *not* tracked by Git**.

Download them from our shared Drive folder:

> https://drive.google.com/drive/folders/1gVmw6qqIGq54gO2mVcW4iRCYqpd38DXn

### Quick download with *gdown*

```bash
pip install -q gdown
gdown --folder --id 1gVmw6qqIGq54gO2mVcW4iRCYqpd38DXn -O external_assets/
