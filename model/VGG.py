import random
from layers import *
import argparse

class VGG11(nn.Module):
    def __init__(self,num_classes=10):
        super(VGG11, self).__init__()
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        #pool = APLayer(2)
        self.layer1 = Layer(3,64,3,1,1)
        self.layer2 = pool
        self.layer3 = Layer(64,128,3,1,1)
        self.layer4 = Layer(128,256,3,1,1)
        self.layer5 = pool
        self.layer6 = Layer(256,512,3,1,1)
        self.layer7 = Layer(512,512,3,1,1)
        self.layer8 = Layer(512,512,3,1,1)
        self.layer9 = pool
        self.layer10 = Layer(512,512,3,1,1)
        self.layer11 = Layer(512,512,3,1,1)

        W = int(32/2/2/2) #cifar10 and cifar100
        #W = 28 #imagenet
        self.linear1 = SeqToANNContainer(nn.Linear(512*W*W,4096))
        self.linear2 = SeqToANNContainer(nn.Linear(4096,4096))
        self.linear3 = SeqToANNContainer(nn.Linear(4096,num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input,args):
        outputs_list = []
        input = add_dimention(input, args.T)
        x = self.layer1(input)
        outputs_list.append(x.clone())
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        outputs_list.append(x.clone())
        
        x = torch.flatten(x, 2)
        x = self.linear1(x) 
        x = self.linear2(x)
        x = self.linear3(x)

        return x, outputs_list

class VGGSNN(nn.Module):
    def __init__(self, num_classes=10):
        super(VGGSNN, self).__init__()
        self.layer1 = Layer(3,64,3,1,1) #cifar10,cifar100
        #self.layer1 = Layer(2,64,3,1,1) #dvscifar10
        self.layer2 = Layer(64,128,3,2,1)
        self.layer3 = Layer(128,256,3,1,1)
        self.layer4 = Layer(256,256,3,2,1)
        self.layer5 = Layer(256,512,3,1,1)
        self.layer6 = Layer(512,512,3,2,1)
        self.layer7 = Layer(512,512,3,1,1)
        self.layer8 = Layer(512,512,3,2,1)

        #W = int(48/2/2/2/2) #dvscifar10
        W = int(32/2/2/2/2) #cifar10,cifar100
        self.classifier = SeqToANNContainer(nn.Linear(512*W*W,num_classes))
        print('num_classes:',num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input,args):
        outputs_list = []
        if args.dataset == 'cifar10':
            input = add_dimention(input, args.T)
        elif args.dataset == 'cifar100':
            input = add_dimention(input, args.T)
        elif args.dataset == 'tinyimagenet':
            input = add_dimention(input, args.T)

        x = self.layer1(input)
        outputs_list.append(x.clone())
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        outputs_list.append(x.clone())
        
        x = torch.flatten(x, 2)
        x = self.classifier(x) 

        return x,outputs_list



if __name__ == '__main__':
    model = VGG11()
    