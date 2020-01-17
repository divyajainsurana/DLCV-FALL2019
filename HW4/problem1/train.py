import os
import numpy as np
import datetime
import matplotlib.pyplot as plt
import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from model import Resnet50
from model import Net
from reader import readShortVideo
from reader import getVideoList
import torch.optim as optim

# loading features extracted by pretrain model
train_features = torch.load('./features/train_features.pt').view(-1,2048)
valid_features = torch.load('./features/valid_features.pt').view(-1,2048)
train_vals = torch.load('./features/train_label.pt').type(torch.LongTensor)
valid_vals = torch.load('./features/valid_label.pt').type(torch.LongTensor)
print("train_features",train_features.shape)
print("train_vals",train_vals.shape)
print("valid_features",valid_features.shape)
print("valid_vals",valid_vals.shape)

# model, optimzer, loss function
feature_size = 2048
model = Net(feature_size).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.5, patience=5, verbose=True) 

loss_function = nn.CrossEntropyLoss()

# some training parameters
BATCH_SIZE = 64
EPOCH_NUM = 500
datalen = len(train_features)
max_accuracy = 0

# start training
model.train()
train_loss = []
valid_acc = []
for epoch in range(EPOCH_NUM):
    print("Epoch:", epoch+1)
    total_loss = 0.0
    total_batchnum = 0

    # shuffle data
    perm_index = torch.randperm(datalen)
    train_features_sfl = train_features[perm_index]
    train_vals_sfl = train_vals[perm_index]

    # training as batches
    for batch_idx, batch_val in enumerate(range(0,datalen ,BATCH_SIZE)):
        if batch_val+BATCH_SIZE > datalen: break
        optimizer.zero_grad()  # zero the parameter gradients
        input_features = train_features_sfl[batch_val:batch_val+BATCH_SIZE]
        input_vals = train_vals_sfl[batch_val:batch_val+BATCH_SIZE]
        input_features = input_features.cuda()
        input_vals = input_vals.cuda()
        # forward + backward + optimize
        predict_labels,_ = model(input_features) #size 64x11
        loss = loss_function(predict_labels, input_vals) #size 64x11 vs 64
        loss.backward()
        optimizer.step()
        total_loss += loss.cpu().data.numpy()
        total_batchnum = batch_idx+1
    print("avg training loss:",total_loss / total_batchnum)
    train_loss.append(total_loss / total_batchnum)

    # validation
    with torch.no_grad():
        model.eval()
        predict_labels,_ = model(valid_features.cuda())
        predict_vals = torch.argmax(predict_labels,1).cpu().data
        accuracy = np.mean((predict_vals == valid_vals).numpy())
        print("validation accuracy: ",accuracy)
        valid_acc.append(accuracy)

    # saving best acc model as best.pth
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        torch.save(model, 'best.pth')
    model.train()
    for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
# plot loss and acc graph
plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.plot(train_loss,color='r')
plt.title("cross entropy training loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.subplot(1,2,2)
plt.plot(valid_acc,color ='r')
plt.title("validation accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.savefig("p1_curve.png")
plt.show()
