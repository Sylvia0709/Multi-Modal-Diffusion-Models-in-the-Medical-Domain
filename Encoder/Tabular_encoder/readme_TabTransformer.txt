1.Overview
This Python script extracts deep learning features from tabular data (stored in CSV files) using the TabTransformer model. It performs preprocessing on the input data, applies the TabTransformer model for feature embedding, and saves the resulting features to a NumPy (.npy) file.

2.Prerequisites

Python 3.7+
The following Python libraries:
pandas
torch (PyTorch)
numpy
scikit-learn
tab-transformer-pytorch
logging (standard library)

3.Setup
download this script；
Install the required dependencies；


4.Run the Script
Place your input CSV file;
Open the Python script and locate the if __name__ == "__main__": section at the bottom.
Update "csv_path =" to the actual path of your input CSV file.

The final extracted feature array will be printed to the console. A NumPy file (.npy) containing the extracted features will be saved in a subdirectory named "result/".
