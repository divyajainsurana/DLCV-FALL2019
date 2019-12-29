import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
# from dataset.data_loader import GetLoader
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Function
from skimage import io, transform
import cv2
import numpy as np
import pandas as pd
import sys
from os import listdir

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        # return class_output, domain_output
        return class_output, domain_output, feature

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class MNIST(Dataset):
    def __init__(self, root_dir, transform=None):
        """ Intialize the MNIST dataset """
        self.root_dir = os.path.join(root_dir)
        # self.landmarks_frame = pd.read_csv(csv_file)
        self.transform = transform
        self.file = sorted(listdir(self.root_dir))
                              
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

def write_csv(predict):
    save_path = sys.argv[3]
    img_name = [str(i).zfill(5)+".png" for i in range(len(predict))]
    csv={'label':predict,'image_name':img_name}
    df = pd.DataFrame(csv,columns=['image_name','label'])
    df.to_csv(os.path.join(save_path),index=0)

def to_csv():
    dataset_name = sys.argv[2]
    print(dataset_name)
    #model_root = os.path.join('.','./save_model/')
    
    cuda = True
    cudnn.benchmark = True
    batch_size = 128
    image_size = 28
    alpha = 0


    if dataset_name == 'svhn':

        dataset =  MNIST(root_dir=os.path.join(sys.argv[1]),transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
        
        my_net = torch.load(os.path.join('mnist_svhn_model_domain2.pth'))
  
    elif dataset_name == 'mnistm':

        dataset =  MNIST(root_dir=os.path.join(sys.argv[1]),transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
        
        my_net = torch.load(os.path.join('svhn__mnistm_model_domain49.pth'))

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0
    predict = []
    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img = data_target

        batch_size = len(t_img)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        # class_label = torch.LongTensor(batch_size)

        if cuda:
            t_img = t_img.cuda()
            # t_label = t_label.cuda()
            input_img = input_img.cuda()
            # class_label = class_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        # class_label.resize_as_(t_label.long()).copy_(t_label.long())

        class_output, _ ,_= my_net(input_data=input_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        # n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()
        
        predict += pred.squeeze().cpu().tolist()
        # n_total += batch_size

        i += 1


    write_csv(predict)

if __name__ == "__main__":
    to_csv()
