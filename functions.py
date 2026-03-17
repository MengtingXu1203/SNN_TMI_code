import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import os
import numpy as np
import logging
from torch.autograd import Variable
import time
from sklearn.metrics import mutual_info_score


def train(model, device, train_loader, criterion, optimizer, epoch, args):
    running_loss = 0
    start_time = time.time()
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(train_loader):
        Mi = 0.0
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.to(device)
        outputs, outputs_list = model(images,args)
        mean_out = outputs.mean(1)

      ## calculate MI
        num_group = len(outputs_list) // 2
        for ng in range(num_group):
            seq1 = torch.mean(outputs_list[2*ng], dim=(3,4)) 
            seq2 = torch.mean(outputs_list[2*ng+1],dim=(3,4))

            seq1 = torch.mean(seq1, dim = 2)*255
            seq2 = torch.mean(seq2, dim = 2)*255
            Mi += batch_mi(seq1,seq2,0,device)


        if args.loss_type=='TET':
            loss = TET_loss(outputs,labels,criterion,args.means,args.lamb)
            if i % 100 == 0:
                print('TET:',loss)
        elif args.loss_type =='CE_MI':
            loss1 = criterion(mean_out,labels)
            loss2 = args.h_lamb * Mi
            if i % 100 == 0:
                print('ce,mi:',loss1,loss2)
            loss = loss1+loss2
        elif args.loss_type =='TET_MI':
            loss1 = TET_loss(outputs,labels,criterion,args.means,args.lamb)
            loss2 = args.h_lamb * Mi
            if i % 100 == 0:
                print('tet,mi:',loss1,loss2)
            loss = loss1+loss2
        else:
            loss = criterion(mean_out,labels)
            if i % 100 == 0:
                print('ce:',loss)
        running_loss += loss.item()
        loss.mean().backward()
        optimizer.step()
        total += float(labels.size(0))
        _, predicted = mean_out.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total

@torch.no_grad()
def test(model, test_loader, device, args):
    correct = 0
    total = 0
    model.eval()
    test_features = []
    test_labels = []
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs, fea = model(inputs, args)
        mean_out = outputs.mean(1)

        _, predicted = mean_out.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        if batch_idx % 100 == 0:
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
    final_acc = 100 * correct / total

    return final_acc

def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    


def TET_loss(outputs, labels, criterion, means, lamb):
    T = outputs.size(1)
    Loss_es = 0
    for t in range(T):
        Loss_es += criterion(outputs[:, t, ...], labels)
    Loss_es = Loss_es / T # L_TET
    if lamb != 0:
        MMDLoss = torch.nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = MMDLoss(outputs, y) # L_mse
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb * Loss_mmd # L_Total
        

