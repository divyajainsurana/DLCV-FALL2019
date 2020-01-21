import torch
import torch.nn as nn
import torch.utils.data
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
import random
import model
import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity
import data
import sys
from tqdm import tqdm
import cv2
import scipy.misc
from matplotlib import cm
import tensorflow as tf
np.random.seed(77)
random.seed(77)
torch.manual_seed(77)
torch.cuda.manual_seed(77)


def get_mse(img_1, img_2):
    img_1 = (img_1*255).astype('uint8')  # change type to 255 to compare with baseline
    img_2 = (img_2*255).astype('uint8')
    return np.mean((img_1 - img_2) ** 2)


def get_ssim(img_1, img_2):
    # Set the RGB channels on the 3rd dim ie: (3, 512, 512) --> (512, 512, 3)
    img_1 = np.swapaxes(img_1, 0, 2)
    img_1 = np.swapaxes(img_1, 0, 1)
    img_2 = np.swapaxes(img_2, 0, 2)
    img_2 = np.swapaxes(img_2, 0, 1)
    return structural_similarity(img_1, img_2, multichannel=True)


def eval(model, epoch):

    dataset = data.DATA(data_dir='Data_Challenge2', mode='test')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False)

    model.eval()
    model.cuda()
    preds = []
    gts = []
    masked_imgs = []
    im =[]
    with torch.no_grad():
        for idx, (masked_img, mask, gt) in enumerate(dataloader):
            masked_img = masked_img.cuda()
            gt = gt.cuda()
            mask = mask.cuda()
            pred = model(masked_img, mask)
            gts.append(gt.squeeze())
            masked_imgs.append(masked_img.squeeze())
            preds.append(pred.squeeze())
            torchvision.utils.save_image(preds[idx], 'output/{}.jpg'.format(idx+401))
        for i in range(len(preds)):
            pred = np.array(preds[i].cpu().detach().numpy())
            pred = (pred*255).astype('uint8')
            pred = np.swapaxes(pred, 0, 2)
            pred = np.swapaxes(pred, 0, 1)
            img = 'Data_Challenge2/test/{}_masked.jpg'.format(401+i)
            img = Image.open(img)
            height,width = img.size
            j = 'output/{}.jpg'.format(401+i)
            jk = cv2.imread(j)
            jk = cv2.resize(jk,(height,width))
            cv2.imwrite('output/{}.jpg'.format(i+401),jk)
             
        mse_total = 0
        ssim_total = 0
        for i in range(len(preds)):
            pred = np.array(preds[i].cpu().detach().numpy())
            gt = gts[i].cpu().detach().numpy()
            mse_total += get_mse(pred, gt)
            ssim_total += get_ssim(pred, gt)
        mse_avg = mse_total / (i + 1)
        ssim_avg = ssim_total / (i + 1)

    return mse_avg, ssim_avg

if __name__ == '__main__':

 if not os.path.exists(sys.argv[1]):
        os.makedirs(sys.argv[1])
 if not os.path.exists(sys.argv[2]):
        os.makedirs(sys.argv[2])
 device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")


 net  = model.Base_Network()


 # fixed seed
 manualSeed = 96
 random.seed(manualSeed)
 torch.manual_seed(manualSeed)

 net.load_state_dict(torch.load('./best_model.pth.tar'))

 dataset = data.DATA(data_dir='Data_Challenge2', mode='test')
 dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2, shuffle=False)

 net.eval()
 net.cuda()
 preds = []
 gts = []
 masked_imgs = []
 im =[]
 with torch.no_grad():
    for idx, (masked_img, mask, gt) in enumerate(dataloader):
        masked_img = masked_img.cuda()
        gt = gt.cuda()
        mask = mask.cuda()
        pred = net(masked_img, mask)
        gts.append(gt.squeeze())
        masked_imgs.append(masked_img.squeeze())
        preds.append(pred.squeeze())

    for i in range(len(preds)):
      pred = np.array(preds[i].cpu().detach().numpy())
      pred = (pred*255).astype('uint8')
      pred = np.swapaxes(pred, 0, 2)
      pred = np.swapaxes(pred, 0, 1)
      im = Image.fromarray(pred,'RGB')
      im.save(os.path.join(sys.argv[2],'{}.jpg'.format(401+i)))
      img = Image.open(os.path.join( sys.argv[1],'{}_masked.jpg'.format(401+i)))
      height,width = img.size
      jk = cv2.imread(os.path.join(sys.argv[2],'{}.jpg'.format(401+i)))
      jk = cv2.resize(jk,(height,width))
      cv2.imwrite(os.path.join(sys.argv[2],'{}.jpg'.format(401+i)),jk)

    mse_total = 0
    ssim_total = 0
    for i in range(len(preds)):
        pred = np.array(preds[i].cpu().detach().numpy())
        gt = gts[i].cpu().detach().numpy()
        mse_total += get_mse(pred, gt)
        ssim_total += get_ssim(pred, gt)
        mse_avg = mse_total / (i + 1)
        ssim_avg = ssim_total / (i + 1)

        info = '\t MSE: %.5f\tSSIM: %.5f' % (mse_avg, ssim_avg)
        print(info)
