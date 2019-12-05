import torch
import torch.nn as nn
import torchvision
import torchvision.models
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import sys
import os

class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()  

        resnet18 = torchvision.models.resnet18(pretrained=True)
        
        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.l1 = resnet18.layer1
        self.l2 = resnet18.layer2
        self.l3 = resnet18.layer3
        self.l4 = resnet18.layer4
        
        self.deconv1 = nn.ConvTranspose2d(512,256,kernel_size=4,stride =2,padding =1,bias=False)
        self.relu1 =nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(256,128,kernel_size=4,stride =2,padding =1,bias=False)
        self.relu2 =nn.ReLU()
        
        self.deconv3 = nn.ConvTranspose2d(128,64,kernel_size=4,stride =2,padding =1,bias=False)
        self.relu3 =nn.ReLU()

        self.deconv4 = nn.ConvTranspose2d(64,32,kernel_size=4,stride =2,padding =1,bias=False)
        self.relu4 =nn.ReLU()

        self.deconv5 = nn.ConvTranspose2d(32,16,kernel_size=4,stride =2,padding =1,bias=False)
        self.relu5 =nn.ReLU()

        self.conv2 = nn.Conv2d(16,9,kernel_size=1, stride=1, padding=0)

    def forward(self, img):
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        img = self.l4(x)
        img = self.relu1(self.deconv1(img))
        img = self.relu2(self.deconv2(img))
        img = self.relu3(self.deconv3(img))
        img = self.relu4(self.deconv4(img))
        img = self.relu5(self.deconv5(img))
        
        img = self.conv2(img)

        return img

class Net_improved(nn.Module):

    def __init__(self, args):
        super(Net_improved, self).__init__()


        ''' declare layers used in this network'''
        #resnet34 = torchvision.models.resnet34(pretrained=True)
        #self.conv1 = resnet34.conv1
        #self.bn1 = resnet34.bn1
        #self.relu = resnet34.relu
        #self.maxpool = resnet34.maxpool
        #self.l1 = resnet34.layer1
        #self.l2 = resnet34.layer2
        #self.l3 = resnet34.layer3
        #self.l4 = resnet34.layer

        # self.trans_conv11 = nn.ConvTranspose2d(2048, 1024, 4, 1, 1, bias=False)  # 11x14 -> 12x13
        # self.relu11 = nn.ReLU()
        #
        # self.trans_conv12 = nn.ConvTranspose2d(1024, 512, 4, 1, 2, bias=False)  # 12x13 -> 11x14
        # self.relu12 = nn.ReLU(_conv1 = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=
        resnet34 = torchvision.models.resnet34(pretrained=True)

        self.conv1 = resnet34.conv1
        self.bn1 = resnet34.bn1
        self.relu = resnet34.relu
        self.maxpool = resnet34.maxpool
        self.l1 = resnet34.layer1
        self.l2 = resnet34.layer2
        self.l3 = resnet34.layer3
        self.l4 = resnet34.layer4
      #  self.l5 = resnet34.layer5
      #  self.l6 = resnet34.layer6

       # self.trans_conv11 = nn.ConvTranspose2d(2048, 1024, 4, 1, 1, bias=False)
       # self.relu6 = nn.ReLU()

       # self.trans_conv12 = nn.ConvTranspose2d(1024, 512, 4, 1, 1, bias=False)
       # self.relu7 = nn.ReLU()

        self.deconv1 = nn.ConvTranspose2d(512,256,kernel_size=4,stride =2,padding =1,bias=False)
        self.relu1 =nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(256,128,kernel_size=4,stride =2,padding =1,bias=False)
        self.relu2 =nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(128,64,kernel_size=4,stride =2,padding =1,bias=False)
        self.relu3 =nn.ReLU()

        self.deconv4 = nn.ConvTranspose2d(64,32,kernel_size=4,stride =2,padding =1,bias=False)
        self.relu4 =nn.ReLU()

        self.deconv5 = nn.ConvTranspose2d(32,16,kernel_size=4,stride =2,padding =1,bias=False)
        self.relu5 =nn.ReLU()

        self.conv2 = nn.Conv2d(16,9,kernel_size=1, stride=1, padding=0)

    def forward(self, img):
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        img = self.l4(x)
        #img = self.l5(x)
        #img = self.l6(x)
        #img = self.relu6(self.trans_conv11(img))
        #img = self.relu7(self.trans_conv12(img))
        img = self.relu1(self.deconv1(img))
        img = self.relu2(self.deconv2(img))
        img = self.relu3(self.deconv3(img))
        img = self.relu4(self.deconv4(img))
        img = self.relu5(self.deconv5(img))

        img = self.conv2(img)

        return img

