
#### Master Thesis: Deep Learning based reconstruction of dynamic non-Cartesian fMRI
This repository contains code and utilities for image reconstruction using deep learning models and k-space downsampling using MATLAB scripts. It also includes MATLAB scripts for evaluating the performance of the test set using various metrics.

###Table of Contents
- DeepLearning
- LS_downsampling
- test_set_evaluation
- Dependencies
- Usage
- Contributing
- License

###DeepLearning
The DeepLearning folder contains Jupyter notebook scripts and Python utilities for training 2D-UNet and 3D-UNet models for image reconstruction. These deep learning models are trained using PyTorch. The scripts in this folder provide detailed instructions on how the models are trained and evaluated on data from the Human Connectome Project (HCP).

##Contents
2D-UNet.ipynb: Jupyter notebook script for training a 2D-UNet model for image reconstruction.
3D-UNet.ipynb: Jupyter notebook script for training a 3D-UNet model for image reconstruction.
utils.py: Python utility functions for data preprocessing, model definition, and training/validation/testing.

###LS_downsampling
The LS_downsampling folder contains MATLAB scripts and functions for k-space downsampling of Looping Star image data. These scripts utilize the k-space trajectory of the Looping Star sequence to downsample the fully sampled k-space of HCP data and reconstruct the downsampled k-space samples. The provided MATLAB scripts demonstrate this process.

##Contents
downsampling_script.m: MATLAB script that demonstrates the usage of downsampling functions.
downsampling_functions.m: MATLAB functions for k-space downsampling based on the LS method.

###test_set_evaluation
The test_set_evaluation folder contains MATLAB scripts and files for evaluating the performance of the trained models on a test set of reconstructed images. These scripts calculate various performance metrics to assess the quality of the reconstructed images. The provided MATLAB scripts guide you through the process of evaluating the test set and generating the performance metrics.

##Contents
evaluation_script.m: MATLAB script that demonstrates the evaluation process and computes performance metrics.
metrics_results.mat: MATLAB file containing the performance metrics results.

###Dependencies
To run the code in this repository, you will need the following dependencies:

PyTorch (version X.X.X)
MATLAB (version X.X)
Make sure to install the required dependencies before executing the scripts.

###Usage
To use the code in this repository, follow these steps:

- Clone the repository to your local machine.
- Set up the required dependencies mentioned in the Dependencies section.
- Navigate to the respective folders (DeepLearning, LS_downsampling, test_set_evaluation) and follow the instructions provided in the individual readme files and script comments.

###Contributing
Contributions to this repository are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request. We appreciate your contributions.
