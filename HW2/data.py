import os
import glob
import torch
import scipy.misc
import random
import numpy
import sys
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image
import skimage.io as imageio
import skimage.transform
import cv2
import pickle
#MEAN = [0.5, 0.5, 0.5]
#STD = [0.5, 0.5, 0.5]
MEAN=[0.485, 0.456, 0.406],
STD=[0.229, 0.224, 0.225]
class DATA(Dataset):
    def __init__(self, args, mode='train'):

        ''' set up basic parameters for dataset '''
        self.mode = mode
        self.data_dir = args.data_dir
        self.img_dir = os.path.join(self.data_dir,self.mode, 'img')
        self.seg_dir = os.path.join(self.data_dir,self.mode, 'seg')

        self.filename = [f for f in os.listdir(self.seg_dir) if f.endswith('.png')]
        #self.filename = [image_basename(f) for f in os.listdir(self.seg_dir) if is_image(f)]
        self.filename.sort()
        ''' set up image transform '''
        if self.mode == 'train':
            self.transform = transforms.Compose([])

        elif self.mode == 'val' or self.mode == 'test':
            self.transform = transforms.Compose([])
    def norm201(self, x):
        x_min = x.reshape(-1, 3).min(axis=0)
        x_max = x.reshape(-1, 3).max(axis=0)
        x = (x - x.min()) / (x.max() - x.min())
        return x

    def __len__(self):
        return len(self.filename)

    def __getitem__(self, idx):

        filename = self.filename[idx]
        if self.mode != 'save' :
            img = cv2.cvtColor(cv2.imread(self.data_dir+ self.mode +'/img/'+ filename, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(float)

       # with open(os.path.join(self.img_dir,filename),'rb') as f:
    #    img = Image.open(f).convert('RGB')
        
       # with open(os.path.join(self.seg_dir,filename),'rb') as f:
          #  seg = Image.open(f).convert('P')
            seg = cv2.imread(self.data_dir+self.mode+'/seg/' +filename, cv2.IMREAD_GRAYSCALE).astype(float)
	        

        img = self.norm201(img)
        img = torch.FloatTensor(img).permute((2,0,1))
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        seg = torch.LongTensor(seg)
        if self.mode == 'save':
            return img
        return img,seg


