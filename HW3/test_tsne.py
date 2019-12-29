import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
# from dataset.data_loader import GetLoader
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
from skimage import io, transform
import cv2
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import manifold, datasets
from sklearn.manifold import TSNE
import seaborn as sns
import random
import torch.nn as nn
from torch.autograd import Function
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

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

        return class_output, domain_output, feature


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
        label = self.landmarks_frame['label'][index]
        label = torch.FloatTensor([label])

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return len(self.landmarks_frame)

np.random.seed(77)
random.seed(77)
torch.manual_seed(77)
torch.cuda.manual_seed(77)


model_root = os.path.join('.','./save_model/')
cuda = True
cudnn.benchmark = True
batch_size = 128
image_size = 28
alpha = 0
predict=[]
features = []
labels = []
#    """load data"""

dataset_SVHN =  MNIST(csv_file="hw3_data/digits/svhn/train.csv", root_dir="hw3_data/digits/svhn/train",transform=transforms.Compose([
                           #    transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
#my_net = torch.load(os.path.join(model_root,'mnist_svhn_model/',"epoch10_mnist2svhn_checkpoint.pth.tar".format(epoch+1)))

dataset_MNIST = MNIST(csv_file="hw3_data/digits/mnistm/train.csv", root_dir="hw3_data/digits/mnistm/train",transform=transforms.Compose([
                           #    transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
torch.cuda.set_device(0)
#my_net = CNNModel()
#if cuda:
#    my_net = my_net.cuda()
my_net = torch.load(os.path.join("save_model/mnist_svhn_model_domain2.pth"))
my_net.eval()
dataloader_svhn = torch.utils.data.DataLoader(
        dataset=dataset_SVHN,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )
dataloader_mnist = torch.utils.data.DataLoader(
        dataset=dataset_MNIST,
        batch_size=batch_size,
        shuffle=False,
)      
#my_net.load_state_dict(net)
#    """ training """

#    my_net = torch.load(os.path.join(model_root,'mnist_svhn_model/',"epoch{}_mnist2svhn_checkpoint.pth.tar".format(epoch+1)))

if cuda:
    my_net = my_net.cuda()



i = 0
n_total = 0
n_correct = 0
feature = []
img_name = []
pre = []
img_name_t = []
domains = []
with torch.no_grad():
    for idx, (img,name) in enumerate(dataloader_svhn):
        img = img.cuda()
        out, _,_ = my_net(img.cuda(),alpha)
           # out = torch.max(out_tar,1)[1]
           # out = out.cpu().numpy()
  #      domain =0
        name = name.long().view(-1).cuda()
        for i in range(len(img)):
            img_name.append(name[i].detach().cpu().numpy().tolist())
            pre.append(out[i].detach().cpu().numpy().tolist())
#            domains.append(domain[i].detach().cpu().numpy().tolist())
    for idx, (img_t,name_t) in enumerate(dataloader_mnist):
        img_t = img_t.cuda()
        out_t, _,_ = my_net(img_t.cuda(),alpha)
           # out = torch.max(out_tar,1)[1]
           # out = out.cpu().numpy()
 #       domain_t = 1
        name_t = name_t.long().view(-1).cuda()
        for i in range(len(img_t)):
            img_name.append(name_t[i].detach().cpu().numpy().tolist())
            pre.append(out_t[i].detach().cpu().numpy().tolist())
   #         domains.append(domain_t[i].detach().cpu().numpy().tolist())
pres= np.array(pre)
tsne = TSNE(n_components=2, init='pca', random_state=501)
X_embedded = tsne.fit_transform(pres) 
color_list = ['red', 'darkgreen', 'black', 'yellow', 'grey', 'orange', 'green', 'pink', 'cyan', 'brown']
sns.set_style('white')
sns.set_palette(['red', 'blue'])
sns.palplot(['red', 'blue'])
plt.figure()
plt.axis('off')
plt.title('SVHN  --> MNIST-M')
print(X_embedded)
#print(len(X_embedded))
#random.shuffle(X_embedded)
data = pd.DataFrame(index=range(len(X_embedded)), columns=['x', 'y', 'label', 'domain'])
for i in range(len(X_embedded)):
    data.loc[i, ['x', 'y']] = X_embedded[i]
        # add group labels
    data.loc[i, ['label']] = img_name[i]
   # data.loc[i, ['domain']] =domains[i]
for i in range(10):
        # add data points
    plt.scatter(x=data.loc[data['label'] == i, 'x'],
                y=data.loc[data['label'] == i, 'y'],
                color=color_list[i],
                alpha=0.20)

        # add label
    plt.annotate(i,
                data.loc[data['label'] == i, ['x', 'y']].mean(),
                horizontalalignment='center',
                verticalalignment='center',
                size=20, weight='bold',
                color='white',
                    backgroundcolor=color_list[i])

plt.show()
facet = sns.lmplot(data=data, x='x', y='y', hue='domain',
                       fit_reg=False, legend=True, legend_out=True, scatter_kws={'alpha': 0.1})
plt.axis('off')
plt.title('SVHN  --> MNIST-M')
plt.show()
    
    



