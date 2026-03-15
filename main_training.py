import argparse
import shutil
import os
import time
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from model.VGG import *
from model.alexnet import *
import data_loaders
from functions import *
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch SNN_TMI')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 10)')
parser.add_argument('--epochs', default=300, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=64, type=int, metavar='N', help='mini-batch size (default: 256), this is the total batch size of all GPUs on the current node when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=0.01, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--seed', default=1000, type=int, help='seed for initializing training. ')
parser.add_argument('--T', '--time', default=4, type=int, metavar='N', help='snn simulation time (default: 2)')
parser.add_argument('--means', default=1.0, type=float, metavar='N', help='make all the potential increment around the means (default: 1.0)')
parser.add_argument('--MI', action="store_true", help='if use MI in loss (default: false)')
parser.add_argument('--lamb', default=0.05, type=float, metavar='N', help='adjust the norm factor to avoid outlier (default: 0.0)')
parser.add_argument('--dataset',default = 'cifar10',type= str,help='datasets,[cifar10,mnist,imagenet,cifar-dvs]')
parser.add_argument('--classes',default = 100,type= int,help='label class,[cifar10:10, cifar100:100]')
parser.add_argument('--model',default = 'VGGSNN',type= str,help='the model used')
parser.add_argument('--gpuids',default = '0',type= str,help='gpuids,0,1,2,3,4,5,6,7')
parser.add_argument('--savemodelpath',default = 'test.pth',type= str,help='savemodelpath')

parser.add_argument('--h_lamb', default=0.05, type=float, help='adjust the H loss')
parser.add_argument('--loss_type', default='CE',type=str, help='loss function')


args = parser.parse_args()

torch.backends.cudnn.enabled = False
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuids
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    seed_all(args.seed)
    if args.dataset == 'cifar10':
        train_dataset, val_dataset = data_loaders.build_cifar()
        args.classes = 10
    elif args.dataset == 'cifar100':
        train_dataset, val_dataset = data_loaders.build_cifar(use_cifar10=False)
        print('cifar100')
        args.classes = 100

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.model == 'VGG11':
        model = VGG11(num_classes = args.classes)
    elif args.model == 'VGGSNN':
        model = VGGSNN(num_classes = args.classes)


    parallel_model = torch.nn.DataParallel(model)
    parallel_model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    best_acc = 0
    best_epoch = 0
    
    
    for epoch in range(args.epochs):
        start_time = time.time()
    
        loss, acc = train(parallel_model, device, train_loader, criterion, optimizer, epoch, args)
        print('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch , args.epochs, loss, acc ))
        print('use time:', time.time()-start_time)
        scheduler.step()
        facc = test(parallel_model, test_loader, device, args)
        print('Epoch:[{}/{}]\t Test acc={:.3f}'.format(epoch , args.epochs, facc ))

        if best_acc < facc:
            best_acc = facc
            best_epoch = epoch + 1
            torch.save(parallel_model.module.state_dict(), args.savemodelpath)
        print('Best Test acc:',best_acc, 'Best epoch:', best_epoch )
        print('\n')