1.Overview
This part provides scripts to train the decoder then use the trained best model for inference on new latent vectors.

2.Prerequisites

Python 3.8+
PyTorch 
einops
pandas
numpy
scikit-learn
matplotlib

3.Setup
download this script；
Install the required dependencies；

4.How to Run Training

Modify these parameters at the beginning of the 'main_train_scnv()'function 

'latent_dir': Path to the directory containing training latent '.pt' files.
'scnv_file_path': Path to the training SCNV data CSV file.
'saved_model_path': Path where the best trained model state dictionary will be saved.
'config_save_path': Path to save the model and training configuration JSON . This is crucial for inference.
'plot_save_path': Path to save the training/validation metrics plot.

Prepare input latent vectors and SCNV CSV file.
Configure the paths and hyperparameters.
Run the script.

5.How to Run Inference
Modify these parameters at the beginning of the 'main_inference_scnv()' function in 'infer_scnv_decoder.py':

'test_latent_dir': Path to the directory containing test latent '.pt' files.
'scnv_test_file_path': Path to the test SCNV data CSV file.
'model_load_path': Path to the trained '.pth'model file.
'config_load_path': Path to the ' model_config.json' file.
'output_folder': Directory where inference results will be saved 'results_plot_path': Full path for saving the inference results plot.
'reconstructed_scnv_output_path': Full path for saving the reconstructed SCNV data as a CSV.

Prepare data.
Configure the paths and hyperparameters.
Run the script.

