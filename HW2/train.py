import os
import torch

import parser
import model
import data
import test

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter
from test import evaluate



def save_model(model, save_path):
    torch.save(model.state_dict(),save_path)    



if __name__=='__main__':

    args = parser.arg_parse()
    
    '''create directory to save trained model and other info'''
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    ''' setup GPU '''
    torch.cuda.set_device(args.gpu)
    
    ''' setup random seed '''
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    ''' load dataset and prepare data loader '''
    print('===> prepare dataloader ...')
    train_loader = torch.utils.data.DataLoader(data.DATA(args, mode='train'),
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=True)
    val_loader   = torch.utils.data.DataLoader(data.DATA(args, mode='val'),
                                               batch_size=args.train_batch, 
                                               num_workers=args.workers,
                                               shuffle=False)
    ''' load model '''
    print('===> prepare model ...') 
    if args.model == 'Net_improved':
        model = model.Net_improved(args)
    elif args.model == 'Net':
        model = model.Net(args)
    else:
        raise Exception('incorrect model name')

    model.cuda() # load model to gpu

    ''' define loss '''
    criterion = nn.CrossEntropyLoss()
    

    ''' setup optimizer '''
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) #weight_decay=args.weight_decay) 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5, patience=5, verbose=True) 
    ''' setup tensorboard '''
    writer = SummaryWriter(os.path.join(args.save_dir, 'train_info'))

    ''' train model '''
    print('===> start training ...')
    iters = 0
    best_mean_iou_score = 0
    running_loss =0
    for epoch in range(1, args.epoch+1):
        
        model.train()
        
        for idx, (img, seg) in enumerate(train_loader):
            
            train_info = 'Epoch: [{0}][{1}/{2}]'.format(epoch, idx+1, len(train_loader))
            iters += 1

            ''' move data to gpu '''
            img, seg = img.cuda(), seg.cuda()
            
            ''' forward path '''
            output = model(img)

            ''' compute loss, backpropagation, update parameters '''
            loss = criterion(output, torch.squeeze(seg.long())) # compute loss
            
            optimizer.zero_grad()         # set grad of all parameters to zero
            loss.backward()               # compute gradient for each parameters
            optimizer.step()              # update parameters

            ''' write out information to tensorboard '''
            writer.add_scalar('loss', loss.data.cpu().numpy(), iters)
            train_info += ' loss: {:.4f}'.format(loss.data.cpu().numpy())

            print(train_info)
        
        if epoch%args.val_epoch == 0:
            ''' evaluate the model '''
            mean_iou_score = evaluate(model, val_loader)        
            writer.add_scalar('mean_iou_score', mean_iou_score, epoch)
            print('Epoch: [{}] Mean IOU Score:{}'.format(epoch, mean_iou_score))
            
            ''' save best model '''
            if mean_iou_score > best_mean_iou_score:
                save_model(model, os.path.join(args.save_dir, 'model_best.pth.tar'))
                best__mean_iou_score= mean_iou_score

            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']


        ''' save model '''
        save_model(model, os.path.join(args.save_dir, 'model_{}.pth.tar'.format(epoch)))
