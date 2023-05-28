
# Master Thesis: Deep Learning based reconstruction of dynamic non-Cartesian fMRI
This repository contains code and utilities for image reconstruction using deep learning models and k-space downsampling using MATLAB scripts. It also includes MATLAB scripts for evaluating the performance of the test set using various metrics.

## Table of Contents
- [DeepLearning](#deeplearning)
- [LS_downsampling](#ls_downsampling)
- [test_set_evaluation](#test_set_evaluation)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Contributing](#contributing)
- [Contact](#contact)

## DeepLearning
The DeepLearning folder contains Jupyter notebook scripts and Python utilities for training 2D-UNet and 3D-UNet models for image reconstruction. These deep learning models are trained using PyTorch. The scripts in this folder demonstrate how the models were trained. Cell outputs are also included

### Contents
dataset_management.py: Dataset classes used for generating the data loaders. \
2D-UNet: Folder containing the utilities used for training a 2D-UNet model as well as a notebook detailing the training process. \
3D-UNet: Folder containing the utilities used for training a 3D-UNet model as well as notebooks detailing the training process on both processed and unprocessed data. This folder also contains the RayTune model selection experiment on processedd data. 

## LS_downsampling
The LS_downsampling folder contains MATLAB scripts and functions for k-space downsampling of Looping Star image data. These scripts utilize the k-space trajectory of the Looping Star sequence to downsample the fully sampled k-space of HCP data and reconstruct the downsampled k-space samples. The provided MATLAB scripts demonstrate this process.

### Contents
downsampling_wDCF.m: MATLAB script that demonstrates the usage of the MIRT non-Uniform k-space tool to downsample a full dataset. The paths in this file are local and needs to be changed to be runnable. A copy of MIRT and k-space coordinates in a MAT file is neccessary. \
downsampling_utils: Folder containing MATLAB functions used during the downsampling, e.g the image squaring, data saving and downsampling functions. \
test_subj.mat: MATLAB file containing the subject IDs of the test set participants.\

## test_set_evaluation
The test_set_evaluation folder contains MATLAB scripts and files for evaluating the performance of the trained models on a test set of reconstructed images. These scripts calculate various performance metrics to assess the quality of the reconstructed images.

### Contents
SnR.m: MATLAB function for calculating the signal-to-noise ration in one volume \
test_metrics.m / test_metrics_proc.m: MATLAB scripts that calculate the metrics for all the subjects in the test set. \
test_metrics_proc.mat / test_metrics_unproc.mat: MATLAB files with the calculated metrics for all test set subjects. \

### Dependencies
To run the code in this repository, you will need the following dependencies:

Python (version 3.9.13)\
PyTorch (version 1.13.0)\
pytorch-3dunet (version 1.3.9 - https://github.com/wolny/pytorch-3dunet) \ 
Pytorch-UNet (https://github.com/milesial/Pytorch-UNet) \
Nibabel (version 4.0.2)\
Ray-Tune (version 1.6.0)\
MATLAB (version 2022b)\
MIRT (https://web.eecs.umich.edu/~fessler/code/)\
The rest of the dependencies can be seen in the environment.yml file. Make sure to install the required dependencies before executing the scripts.

### Usage
The scripts in this repository are the training scripts used during this master thesis. With the dependencies above, all functions and classes should be usable. If you want to use any of them, follow these steps:

- Set up the required dependencies mentioned in the Dependencies section.
- Clone the repository to your local machine.
- Follow the procedures from the jupyter/MATLAB scripts for a demonstration of how to use the functions

### Contributing
Contributions to this repository are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request. I appreciate your contributions.

### Contact
For access to trained models, or comments and input concerning the repository can be directed at: mariusrusaas@gmail.com
