import argparse
import shutil
import os
import time
import torch
import warnings
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.autograd import Variable
from model.VGG import *
from model.alexnet import *
import data_loaders
from functions import *

import os.path as osp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.optim import lr_scheduler



parser = argparse.ArgumentParser(description='PyTorch SNN_TMI')
parser.add_argument('-j','--workers',default=16,type=int,metavar='N',help='number of data loading workers (default: 10)')
parser.add_argument('--epochs',default=150,type=int,metavar='N',help='number of total epochs to run')
parser.add_argument('--start_epoch',default=0,type=int,metavar='N',help='manual epoch number (useful on restarts)')
parser.add_argument('-b','--batch_size',default=128,type=int,metavar='N',help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr','--learning_rate',default=0.01,type=float,metavar='LR',help='initial learning rate',dest='lr')
parser.add_argument('--seed',default=1000,type=int,help='seed for initializing training. ')
parser.add_argument('--T',default=4,type=int,metavar='N',help='snn simulation time (default: 2)')
parser.add_argument('--means',default=1.0,type=float,metavar='N',help='make all the potential increment around the means (default: 1.0)')
parser.add_argument('--TET',action="store_true", help='if use Temporal Efficient Training (default: false)')
parser.add_argument('--lamb',default=1e-3,type=float,metavar='N',help='adjust the norm factor to avoid outlier (default: 0.0)')
parser.add_argument('--dataset',default = 'cifar10',type= str,help='datasets,[cifar10,mnist,imagenet,cifar-dvs]')
parser.add_argument('--classes',default = 100,type= int,help='label class,[cifar10:10, cifar100:100]')
parser.add_argument('--model',default = 'VGGSNN',type= str,help='the model used')
parser.add_argument('--modelpath',default = 'test.pth',type= str,help='load model modelpath')
## added for pgd attack
parser.add_argument('--epsilon', default=4.0/255, type=float, help='pgd attack epsilon')
parser.add_argument('--adv_steps', default=4, type=int, help='pgd attack steps')
parser.add_argument('--step_size', default=1.0/255, type=float, help='pgd attack step size')

parser.add_argument('--gpuids',default = '0',type= str,help='gpuids,0,1,2,3,4,5,6,7')



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
    elif args.dataset == 'dvscifar10':
        train_dataset, val_dataset = data_loaders.build_dvscifar('/home/datasets/cifar10dvs') # change to your path
        print('dvscifar10')
        args.classes = 10
    elif args.dataset == 'tinyimagenet':
        train_dataset, val_dataset = data_loaders.build_tiny_imagenet()
        print('tiny_imagenet')
        args.classes = 200
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)


    #model = VGGSNNwoAP()
    if args.model == 'VGG11':
        model = VGG11(num_classes = args.classes)
    elif args.model == 'VGGSNN':
        model = VGGSNN(num_classes = args.classes)
    elif args.model == "alexnet":
        model = AlexNet(num_classes = args.classes)

    print(args.modelpath)
    state_dict = torch.load(args.modelpath, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
    
    
    parallel_model = torch.nn.DataParallel(model)
    parallel_model.to(device)

    # natural
    facc = test (parallel_model, test_loader, device, args)
    print('Test Accuracy of the model: %.3f' % facc)

    #fgsm
    fgsm_facc = test_fgsm(parallel_model, test_loader, device, args)
    print('fgsm Test Accuracy of the model: %.3f' % fgsm_facc)

    #pgd
    adv_facc = test_adv(parallel_model, test_loader, device, args)
    print('pgd Test Accuracy of the model: %.3f' % adv_facc)



