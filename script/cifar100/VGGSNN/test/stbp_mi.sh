#!/bin/bash


python main_test.py \
    --dataset cifar100 \
    --model VGGSNN \
    --modelpath modelpath/cifar100/VGGSNN/stbp_mi.pth \
    --epsilon 0.015686 \
    --adv_steps 4 \
    --step_size 0.0039 \
    --gpuids 0 
    