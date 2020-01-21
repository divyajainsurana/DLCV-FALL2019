import torch
import torch.nn as nn
import torch.utils.data
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
import random
import model
import data
from test import eval
from tqdm import tqdm


np.random.seed(999)
random.seed(999)
torch.manual_seed(999)
torch.cuda.manual_seed(999)

batch_size = 1
nb_epoch = 100
if __name__ == '__main__':

    if not os.path.exists('output'):
        os.makedirs('output')
    if not os.path.exists('models'):
        os.makedirs('models')

    print("Loading dataset...")
    dataset = data.DATA(data_dir='Data_Challenge2', mode='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True)

    net = model.Base_Network()
    net.cuda()

    criterion = nn.MSELoss()
    criterion.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001,betas =(0.5,0.999))

    # scheduler change lr at every milestones reached
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,70], gamma=0.1)

    max_score = 0
    # checkpoint = torch.load('models/model_99.pth.tar')
    # G.load_state_dict(checkpoint)
    print('Starting training...')

    for epoch in tqdm(range(nb_epoch)):

        for idx, (masked_img, mask, gt) in enumerate(dataloader):
            optimizer.zero_grad()
            masked_img = masked_img.cuda()
            mask = mask.cuda()
            gt = gt.cuda()

            pred = net(masked_img, mask)

            loss = criterion(pred,gt)
            loss.backward()
            optimizer.step()

        mse, ssim = eval(net, epoch)

      #  torchvision.utils.save_image([masked_img[0], pred[0], gt[0]], 'output/{}_train.jpg'.format(epoch))

        # printing training info
        train_info = 'Epoch: [%i]\tMSE: %.5f\tSSIM: %.5f\tloss: %.4f\tLR:%.5f' % (epoch, mse, ssim, loss, scheduler.get_lr()[0])
        print(train_info)

        scheduler.step()
        score = 1 - mse / 100 + ssim
        # Save model
        if score > max_score:
            torch.save(net.state_dict(), 'models/best_model.pth.tar')
            max_score = score
