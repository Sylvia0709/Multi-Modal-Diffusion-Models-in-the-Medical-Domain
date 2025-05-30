# Multi-Modal Diffusion Model for Medical Data Completion

This project implements a multi-modal diffusion model to handle heterogeneous clinical and imaging data for cancer patients. The goal is to predict missing modalities using observed data and generate synthetic latent representations.

## Project Structure

- `main.py`: The main training and inference script.
- `data_loader.py`: Data loading and preprocessing logic.
- `model/`: Contains model components such as UNet, attention fusion, and conditioning modules.
- `train_generate.py`: Functions for training, evaluation, and generation.

## Modalities

- Clinical
- Mutation
- Methylation
- SCNV
- CT images
- MR images

## Training

The model trains a diffusion process conditioned on input modalities and cancer types, using UNet and cross-attention fusion layers.

## Requirements

Install required packages with:

```bash
pip install -r requirements.txt
```

## How to Run

1. Add the pro72_data folder shortcut to your 'My Drive' as expected.
2. Run the script:

```bash
python main.py
```

Make sure to configure paths correctly if you're not using Google Colab.

## Output

- Trained weights in `checkpoints/`
- Latent vectors saved for each patient in the specified output folder
- Evaluation results printed to console

## License

For academic use only.
