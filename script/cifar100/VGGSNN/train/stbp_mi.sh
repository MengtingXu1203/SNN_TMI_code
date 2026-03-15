#!/bin/bash
# --savemodelpath (change to your own path)

python main_training.py \
    --dataset cifar100 \
    --model VGGSNN \
    --loss_type CE_MI \
    --h_lamb 0.05 \
    --epochs 300 \
    --savemodelpath modelpath/cifar100/VGGSNN/stbp_mi.pth \
    --gpuids 0 
    