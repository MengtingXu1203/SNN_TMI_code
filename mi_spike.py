import torch
from sklearn.metrics import mutual_info_score
import torch.nn as nn
import os
import numpy as np




epsilon = 1e-10

def marginalPdf(values, device,min_num=0,max_num =1, num_bins=2):
    sigma = 0.4
    sigma = 2*sigma**2
    bins = nn.Parameter(torch.linspace(min_num, max_num, num_bins, device=device).float(), requires_grad=False)
    residuals = values - bins.unsqueeze(0).unsqueeze(0)
    kernel_values = torch.exp(-0.5*(residuals / sigma).pow(2))

    pdf = torch.mean(kernel_values, dim=1)
    normalization = torch.sum(pdf, dim=1).unsqueeze(1) + epsilon
    pdf = pdf / normalization

    return pdf, kernel_values

def jointPdf(kernel_values1, kernel_values2):
    joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2) 
    normalization = torch.sum(joint_kernel_values, dim=(1,2)).view(-1, 1, 1) + epsilon
    pdf = joint_kernel_values / normalization

    return pdf

def batch_mi(seq1, seq2,t,device):
    seq1 = seq1.unsqueeze(2).to(device)
    seq2 = seq2.unsqueeze(2).to(device)

    pdf_x1, kernel_values1 = marginalPdf(seq1, device, min_num =0,max_num =255, num_bins=256)
    pdf_x2, kernel_values2 = marginalPdf(seq2, device, min_num =0,max_num =255, num_bins=256)
    pdf_x1x2 = jointPdf(kernel_values1, kernel_values2)

    H_x1 = -torch.sum(pdf_x1*torch.log2(pdf_x1 + epsilon), dim=1)
    H_x2 = -torch.sum(pdf_x2*torch.log2(pdf_x2 + epsilon), dim=1)
    H_x1x2 = -torch.sum(pdf_x1x2*torch.log2(pdf_x1x2 + epsilon), dim=(1,2))

    mutual_information = H_x1 + H_x2 - H_x1x2

    mutual_information = torch.mean(mutual_information)

    return mutual_information



