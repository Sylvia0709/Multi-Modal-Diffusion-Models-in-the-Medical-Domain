1.Overview
This Python script is designed to extract deep learning features from 3D medical images (stored as .npz`files). 

2.Prerequisites
Python 3.8+
PyTorch (version compatible with MONAI, e.g., 1.10+ or as per MONAI's latest recommendations)
MONAI (e.g., 0.8.0+ or latest stable version)
NumPy

3.Setup
download this script；
Install the required dependencies；

4.Run the Script
Place your input npz file;
Open the Python script and locate the if __name__ == "__main__": 
Update "data_dir="  and "feature_save_dir=" to the actual path 

Extracted features for each successfully processed input image are saved as individual PyTorch tensor files (.pt) in the directory specified by "feature_save_dir=" 