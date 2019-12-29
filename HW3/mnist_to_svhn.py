import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
# from dataset.data_loader import GetLoader
from torchvision import datasets
from torchvision import transforms
#from functions import ReverseLayerF
from torch.autograd import Function
from sklearn import manifold, datasets

#from model import CNNModel
import numpy as np
from test import test
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from skimage import io, transform
import cv2
import pandas as pd


class MNIST(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """ Intialize the MNIST dataset """
        self.root_dir = root_dir
        self.landmarks_frame = pd.read_csv(csv_file)
        self.transform = transform
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        img_name = os.path.join(self.root_dir,self.landmarks_frame.iloc[index,0])
        image = io.imread(img_name)
        #image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        label = self.landmarks_frame['label'][index]
        label = torch.FloatTensor([label])

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.landmarks_frame)

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

source_dataset_name = 'MNIST'
target_dataset_name = 'SVHN'
# source_image_root = os.path.join('..', 'dataset', source_dataset_name)
# target_image_root = os.path.join('..', 'dataset', target_dataset_name)
# model_root = os.path.join('..', 'models')
cuda = True
cudnn.benchmark = True
device = torch.device("cuda:0" if (torch.cuda.is_available() and cuda == True) else "cpu")

lr = 0.0001
batch_size = 128
image_size = 28
n_epoch = 1

manual_seed =  random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
sourceTrainDataset = MNIST(csv_file="hw3_data/digits/mnistm/train.csv", root_dir="hw3_data/digits/mnistm/train",transform=transforms.Compose([
        #                     transforms.Resize(image_size),  
                             transforms.ToTensor(),
                               transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                           ]))
dataloader_source = torch.utils.data.DataLoader(dataset=sourceTrainDataset, batch_size=batch_size, shuffle=True, num_workers =8)

my_net = CNNModel()

# setup optimizer

optimizer = optim.Adam(my_net.parameters(), lr=lr)

loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()

if (device.type == 'cuda') and (cuda == True):
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in my_net.parameters():
    p.requires_grad = True

# training
features = []

for epoch in range(n_epoch):

    len_dataloader = len(dataloader_source)
    data_source_iter = iter(dataloader_source)
    

    i = 0
    while i < len_dataloader:

        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        data_source = data_source_iter.next()
        s_img, s_label = data_source

        my_net.zero_grad()
        batch_size = len(s_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)
        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.long()

        if (device.type == 'cuda') and (cuda == True):
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(s_img).copy_(s_img)
        class_label.resize_as_(s_label.long()).copy_(s_label.long())

        class_output, domain_output,feature= my_net(input_data=input_img, alpha=alpha) 
        features.append(feature.detach().cpu().numpy().tolist())
        err_s_label = loss_class(class_output, class_label.squeeze())
        err_s_domain = loss_domain(domain_output, domain_label)

        
        err = err_s_domain + err_s_label
        err.backward()
        optimizer.step()

        i += 1

 #       print ('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
  #            % (epoch, i, len_dataloader, err_s_label.cpu().data.numpy(),
   #              err_s_domain.cpu().data.numpy(), err_t_domain.cpu().data.numpy()))

    #torch.save(my_net, './save_model/mnist_svhn_model'+str(epoch)+'.pth')
#    print(len(features))
    out = np.asarray(features[0])
    out=np.squeeze(out)
#    tsne = manifold.TSNE(n_components=2)
#    X_tsne = tsne.fit_transform(out)
    filename = os.path.join('./save_model/mnist_svhn_model/',"epoch{}_mnist2svhn_checkpoint.pth.tar".format(epoch+1))
    torch.save(my_net,f = filename)
#    test(source_dataset_name, epoch)
    test(target_dataset_name, epoch)

print ('done')
tsne = manifold.TSNE(n_components=2)
X_tsne = tsne.fit_transform(out)
