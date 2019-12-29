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

def test(dataset_name, epoch):
    assert dataset_name in ['MNIST', 'SVHN']

    model_root = os.path.join('.','./save_model/')
    cuda = True
    cudnn.benchmark = True
    batch_size = 128
    image_size = 28
    alpha = 0
    predict=[]
    features = []
    labels = []
    """load data"""

    if dataset_name == 'SVHN':
        dataset =  MNIST(csv_file="hw3_data/digits/svhn/test.csv", root_dir="hw3_data/digits/svhn/test",transform=transforms.Compose([
                           #    transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
        my_net = torch.load(os.path.join(model_root,'mnist_svhn_model/',"epoch{}_mnist2svhn_checkpoint.pth.tar".format(epoch+1)))
    else:
        dataset = MNIST(csv_file="hw3_data/digits/mnistm/test.csv", root_dir="hw3_data/digits/mnistm/test",transform=transforms.Compose([
                           #    transforms.Resize(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
        my_net = torch.load(os.path.join(model_root,'mnist_svhn_model/',"epoch{}_svhn2mnist_checkpoint.pth.tar".format(epoch+1)))
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8
    )

    """ training """

#    my_net = torch.load(os.path.join(model_root,'mnist_svhn_model/',"epoch{}_mnist2svhn_checkpoint.pth.tar".format(epoch+1)))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0
    feature = []
    img_name = []
    pre = []
    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)

        if cuda:
            t_img = t_img.cuda()
            t_label = t_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        class_label.resize_as_(t_label.long()).copy_(t_label.long())

        class_output, domain_output,feature= my_net(input_data=input_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1] 
        n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()
        predict += pred.squeeze().cpu().tolist()
        n_total += batch_size

        i += 1
    
    
    
    accu = n_correct.data.numpy() * 100.0 / n_total
    print ('epoch: %d, accuracy of the %s dataset: %f' % (epoch, dataset_name, accu))
    if (accu > 25):
        write_csv(predict)
        
    

def write_csv(predict):
    img_name = [str(i).zfill(5)+".png" for i in range(len(predict))]
    csv={'label':predict,'image_name':img_name}
    df = pd.DataFrame(csv,columns=['image_name','label'])
    df.to_csv("./save_model/predict_svhn_to_mnist.csv",index=0)

