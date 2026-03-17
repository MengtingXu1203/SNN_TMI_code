import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder, MNIST, ImageNet
import warnings
import os
import torchvision
from os import listdir
import numpy as np
import time
from os.path import isfile, join

import glob
from shutil import move
from os import rmdir

warnings.filterwarnings('ignore')


def build_cifar(cutout=False, use_cifar10=True, download=False):
    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    aug.append(transforms.ToTensor())

    if cutout:
        aug.append(cutout(n_holes=1, length=16))

    if use_cifar10:
        aug.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root='/home/datasets/',
                                train=True, download=download, transform=transform_train)
        val_dataset = CIFAR10(root='/home/datasets/',
                              train=False, download=download, transform=transform_test)

    else:
        aug.append(
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root='/home/datasets/',
                                 train=True, download=download, transform=transform_train)
        val_dataset = CIFAR100(root='/home/datasets/',
                               train=False, download=download, transform=transform_test)

    return train_dataset, val_dataset

def build_dvscifar(path):
    train_path = path + '/train'
    val_path = path + '/test'
    print(path)
    train_dataset = DVSCifar10(root=train_path)
    val_dataset = DVSCifar10(root=val_path)

    return train_dataset, val_dataset

def build_tiny_imagenet():
    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2770, 0.2691, 0.2821])
    root = '/home/datasets/tiny-imagenet-200'
    train_root = os.path.join(root,'train')
    val_root = os.path.join(root,'val')
    train_dataset = ImageFolder(
        train_root,
        transforms.Compose([
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    )
    val_dataset = ImageFolder(
        val_root,
        transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            normalize,
        ])
    )
    return train_dataset, val_dataset




if __name__ == '__main__':
    train_set, test_set = build_tiny_imagenet()
  

