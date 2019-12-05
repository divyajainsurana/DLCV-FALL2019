

import numpy as np
import sys
import os
from PIL import Image
import cv2
import model
import pickle
from argparse import ArgumentParser


import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as trns
from mean_iou_evaluate import mean_iou_score

class ImageDataset(Dataset):

    def __init__(self, data_dir, transform):

        self.data_dir = data_dir
        self.img_files = None
        self.initFile()

        self.transform = transform

    def initFile(self):
        fn = os.listdir(self.data_dir)
        self.img_files = sorted(fn, key=lambda x: int(x[:x.index('.')]))

    def norm201(self, x):
        x_min = x.reshape(-1, 3).min(axis=0)
        x_max = x.reshape(-1, 3).max(axis=0)
        x = (x - x.min()) / (x.max() - x.min())
        return x

    def __getitem__(self, index):

        fn = self.img_files[index]
        # x = Image.open(self.data_dir + '/img/' + fn).convert('RGB')
        x = cv2.cvtColor(cv2.imread(self.data_dir + fn,
                                    cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(float)
        norm_x = self.norm201(x)
        x_ten = torch.FloatTensor(norm_x).permute((2, 0, 1))
        x_ten = trns.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x_ten)

        # normalize to 0-1
        # trans_x = self.transform(x)

        return x_ten

    def __len__(self):
        return len(self.img_files)


def predict(model, valid_data):

    model = model.cuda()
    model = model.eval()

    preds = []

    with torch.no_grad():
        for img in valid_data:
            batch_size = img.shape[0]

            img = img.cuda()

            out = model(img)
            pred = out.argmax(dim=1)

            preds.append(pred.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    model = model.cpu()

    return preds
def evaluate(model, data_loader, save=False):
    """ set model to evaluate mode """
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():  # do not need to caculate information for gradient during eval
        for idx, (imgs, gt) in enumerate(data_loader):
            imgs = imgs.cuda()
            pred = model(imgs)

            _, pred = torch.max(pred, dim=1)

            pred = pred.cpu().numpy().squeeze()
            gt = gt.numpy().squeeze()

            preds.append(pred)
            gts.append(gt)

    gts = np.concatenate(gts)
    preds = np.concatenate(preds)

    # ''' add predictions to output_dir'''
    # if save:
    #     output_dir = args.save_dir
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
    #     for idx, pred in enumerate(preds):
    #         im = Image.fromarray(np.uint8(pred))
    #         save_path = os.path.join(output_dir, f"{idx:04}.png")
    #         im.save(save_path)

    return mean_iou_score(gts, preds)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("source_dir")
    parser.add_argument("pred_dir")
    parser.add_argument("model_path")
    parser.add_argument("model")

    args = parser.parse_args()

    dataset_transform = trns.Compose([
        # trns.RandomRotation([0, 360]),
        trns.RandomHorizontalFlip(),
        # trns.ToTensor(),
        # trns.Normalize(mean=[], std=[])
    ])

    test_dataset = ImageDataset(args.source_dir + '/', dataset_transform)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=32,
                                 shuffle=False,
                                 num_workers=4
                                 )
    
    if args.model == 'Net_improved':
        model = model.Net_improved(args).cuda()
    elif args.model == 'Net':
        model = model.Net(args).cuda()
    else:
        raise Exception('incorrect model name')
    #model = model.Net(args)
    model.load_state_dict(torch.load(args.model_path))
    preds = predict(model, test_dataloader)
 #   print('Testing iou score: {}'.format(preds))

    for idx, fn in enumerate(test_dataset.img_files):
        img = Image.fromarray(preds[idx].astype(np.uint8))
        img.save(args.pred_dir + '/' + fn)
        # scipy.toimage(preds[idx], cmin=0, cmax=8).save(args.pred_dir + '/' + fn)
