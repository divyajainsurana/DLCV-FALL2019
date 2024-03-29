import torch
import torch.cuda as tcuda
import torchvision.utils as tutils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from skimage import io, transform
import math
import pandas as pd
import os
from tqdm import tqdm
import cv2
import sys
from os import listdir

numEpochs = 25
batchSize = 128
learningRate = 0.0005
weightDecay = 2.5e-4
rgb2grayWeights = [0.2989, 0.5870, 0.1140]

class MNIST(Dataset):
    def __init__(self, root_dir, transform=None):
        """ Intialize the MNIST dataset """
        self.root_dir = root_dir
        # self.landmarks_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.file = sorted(listdir(root_dir))
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        img_name = os.path.join(self.root_dir,self.file[index])
        image = io.imread(img_name)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # label = self.landmarks_frame['label'][index]
        # label = torch.FloatTensor([label])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.file)
class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 20 , 5)
        self.bn1 = nn.BatchNorm2d(20)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.bn2 = nn.BatchNorm2d(50)
        self.fc1 = nn.Linear(50 * 4 * 4, 500)
        self.bn3  = nn.BatchNorm1d(500)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = F.leaky_relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.leaky_relu(self.bn3(self.fc1(out)))
        return out

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        return self.fc2(x)

def to_csv(predict):
    save_path = sys.argv[3]
    img_name = [str(i).zfill(5)+".png" for i in range(len(predict))]
    csv={'label':predict,'image_name':img_name}
    df = pd.DataFrame(csv,columns=['image_name','label'])
    df.to_csv(os.path.join(save_path),index=0)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(500, numberHiddenUnitsD)
        self.fc2 = nn.Linear(numberHiddenUnitsD, numberHiddenUnitsD)
        self.fc3 = nn.Linear(numberHiddenUnitsD, 2)
        self.bn1 = nn.BatchNorm1d(numberHiddenUnitsD)
        self.bn2 = nn.BatchNorm1d(numberHiddenUnitsD)
        
    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.fc1(x)))
        out = F.leaky_relu(self.bn2(self.fc2(out)))
        return self.fc3(out)

dataset_name = sys.argv[2]
if dataset_name == 'svhn':

    model = torch.load("targetTrainedModelMNISTtoSVHN_improved")
    classifier = torch.load("classifier_2")


    test_dataset = MNIST(root_dir=sys.argv[1],transform=transforms.Compose([

                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batchSize, shuffle=False)

elif dataset_name == 'mnistm':
    
    model = torch.load("targetTrainedModelSVHNtoMNIST_improved")
    classifier = torch.load("classifier")

    test_dataset = MNIST(root_dir=sys.argv[1],transform=transforms.Compose([

                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batchSize, shuffle=False)        


model.eval() 
correctClass = torch.zeros(10)
totalClass = torch.zeros(10)
pred = []
for images in test_loader:

    if tcuda.is_available():
        
        images = images.cuda()

    # labels[torch.eq(labels, 10)] = 0
    # labels = torch.squeeze(labels).long()
    if images.size(1) == 3:
        images= rgb2grayWeights[0] * images[:, 0, :, :] + rgb2grayWeights[1] * images[:, 1, :, :] + \
                       rgb2grayWeights[2] * images[:, 2, :, :]
        images.unsqueeze_(1)

    images = Variable(images)
    outputs = classifier(model(images))
    _, predicted = torch.max(outputs.data, 1)
    pred += predicted.cpu().tolist()
    
  
to_csv(pred)
