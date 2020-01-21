import os
import json
import torch
import scipy.misc
import numpy

import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from PIL import Image

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def load_image(file):
    return Image.open(file)


class DATA(Dataset):
    def __init__(self, data_dir, mode='train'):

        """ set up basic parameters for dataset """
        self.mode = mode
        self.data_dir = data_dir
        if self.mode == 'train':
            self.img_dir = os.path.join(self.data_dir, 'train')
            self.gt_dir = os.path.join(self.data_dir, 'train_gt')
        elif self.mode == 'test':
            self.img_dir = os.path.join(self.data_dir, 'test')
            self.gt_dir = os.path.join(self.data_dir, 'test_gt')

        ''' set up list of filenames for retrieval purposes'''
        self.filenames = [image_basename(f) for f in os.listdir(self.img_dir)]
        self.filenames.sort()
        self.gt_names = [image_basename(f) for f in os.listdir(self.gt_dir)]
        self.gt_names.sort()

        ''' set up image transform '''
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((1024, 1024)),
               # transforms.RandomHorizontalFlip(),
               # transforms.CenterCrop((512, 512)),
                transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
               #  transforms.Normalize(MEAN, STD)
            ])
            self.mask_transform = transforms.Compose([
        #        transforms.Pad(padding =50, fill =0),
                transforms.Resize((1024, 1024)),
                
               # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            ])
            self.gt_transform = transforms.Compose([
                transforms.Resize((1024,1024)),
                transforms.ToTensor(),
            ])

        elif self.mode == 'test':
            self.transform = transforms.Compose([
#                transforms.Pad(padding =50, fill =1),
                transforms.Resize((1024, 1024)),
                # /!\ to remove later
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                # transforms.Normalize(MEAN, STD)
            ])
            self.mask_transform = transforms.Compose([
              #  transforms.Pad(padding =50, fill =1),
                transforms.Resize((1024, 1024)),  # /!\ to remove later
               
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
            ])

            self.gt_transform = transforms.Compose([
                transforms.Resize((1024,1024)),
                transforms.ToTensor()
            ])


    def __len__(self):
        return len(self.gt_names)

    def __getitem__(self, idx):

        """ get data """
        masked_name = self.filenames[2 * idx + 1]  # filenames includes both mask and masked images
        mask_name = self.filenames[2 * idx]
        gt_name = self.gt_names[idx]
        with open(os.path.join(self.img_dir, masked_name + '.jpg'), 'rb') as f:
            img = load_image(f).convert('RGB')
        #    width, height = img.size
        #    print("The dimension of the image is", width, "x", height)
        with open(os.path.join(self.img_dir, mask_name + '.jpg'), 'rb') as f:
            mask = load_image(f).convert('P')
        with open(os.path.join(self.gt_dir, gt_name + '.jpg'), 'rb') as f:
            gt = load_image(f).convert('RGB')


        # data_aug = False
        # if self.mode == 'train':
        #     data_aug = True
        #     import imgaug as ia
        #     import imgaug.augmenters as iaa
        #
        #     ia.seed(1)
        #
        #     seq = iaa.Sequential([
        #         iaa.Add((-30, 30), per_channel=True),  # change channel value
        #         iaa.Fliplr(0.5),  # flip horizontally with probability 0.5
        #         iaa.Crop(percent=(0, 0.1))  # crop between 0 and 10%, keeping the same size
        #         # iaa.SomeOf((0, 3),  # only apply some of the transformations
        #         # iaa.GaussianBlur(sigma=(0, 2.0)),
        #         # iaa.Dropout((0.05, 0.1)),  # drop 5% or 20% of all pixels
        #         # iaa.Sharpen((0.0, 0.5)),  # sharpen the image
        #         # iaa.Affine(rotate=(-20, 20)),  # rotate by -20 to 20 degrees (affects segmaps)
        #         # iaa.ElasticTransformation(alpha=(0, 30), sigma=5),  # apply water effect (affects segmaps)
        #         # iaa.PerspectiveTransform(scale=(0.01, 0.1))
        #         # )
        #
        #     ], random_order=True)
        #
        #     img = numpy.array(img)
        #     seg = numpy.array(seg)
        #
        #     seg = ia.SegmentationMapsOnImage(seg, shape=img.shape)
        #
        #     img, seg = seq(image=img, segmentation_maps=seg)
        #     img = Image.fromarray(img.astype('uint8'), 'RGB')
        #     seg = ia.SegmentationMapsOnImage.get_arr(seg)

        ''' transform image '''
        if self.transform is not None:
            img = self.transform(img)
            mask = self.mask_transform(mask)
            gt = self.gt_transform(gt)

        return img, mask, gt
