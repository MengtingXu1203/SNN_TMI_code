# SNN_TMI
Code for "Robust Spiking Neural Networks by Temporal Mutual Information (CVPR2026)"

## Prerequisites

The Following Setup is tested and it is working:

- Python>=3.5

- Pytorch>=1.9.0

- Cuda>=10.2

## Description

- Use a triangle-like surrogate gradient `ZIF` in `layers.py` for step function forward and backward.

- It's very easy to build snn convolution layer by `Layer` in `layers.py`.

   `self.conv = nn.Sequential(Layer(2,64,3,1,1),Layer(64,128,3,1,1),)`

- The 0-th and 1-th dimension of snn layer's input and output are batch-dimension and time-dimension.

- Mutual information calculation is in `mi_spike.py`

## Training \& Testing

- CIFAR100, VGGSNN
   + STBP_TMI. train: run `bash script/cifar100/VGGSNN/train/stbp_mi.sh`; test: run `bash script/cifar100/VGGSNN/test/stbp_mi.sh`
      
   + TET_TMI. train: run `bash script/cifar100/VGGSNN/train/tet_mi.sh`; test: run `bash script/cifar100/VGGSNN/test/tet_mi.sh`

