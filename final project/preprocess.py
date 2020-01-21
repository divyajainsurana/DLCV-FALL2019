from PIL import Image
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, Resize, RandomHorizontalFlip
import torch
from torch.utils.data import Dataset
from os import listdir, walk
from os.path import join
from random import randint

def CheckImageFile(filename):
    return any(filename.endswith(extention) for extention in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP'])

def ImageTransform(loadSize, cropSize):
    return Compose([
        Resize(size=loadSize, interpolation=Image.BICUBIC),
        RandomCrop(size=cropSize),
        RandomHorizontalFlip(p=0.5),
        ToTensor(),
    ])

def MaskTransform(cropSize):
    return Compose([
        Resize(size=cropSize, interpolation=Image.NEAREST),
        ToTensor(),
    ])


def PairedImageTransform(cropSize):
    return Compose([
        Resize(size=cropSize, interpolation=Image.NEAREST),
        ToTensor(),
    ])

class Data(Dataset):
    def __init__(self, dataRoot, maskRoot, loadSize, cropSize):
        super(GetData, self).__init__()

        self.imageFiles = [join (dataRootK, files) for dataRootK, dn, filenames in walk(dataRoot) \
            for files in filenames if CheckImageFile(files)]
        self.masks = [join (dataRootK, files) for dataRootK, dn, filenames in walk(maskRoot) \
            for files in filenames if CheckImageFile(files)]
        self.numOfMasks = len(self.masks)
        self.loadSize = loadSize
        self.cropSize = cropSize
        self.ImgTrans = ImageTransform(loadSize, cropSize)
        self.maskTrans = MaskTransform(cropSize)
    
    def __getitem__(self, index):
        img = Image.open(self.imageFiles[index])
        mask = Image.open(self.masks[randint(0, self.numOfMasks - 1)])

        groundTruth = self.ImgTrans(img.convert('RGB'))
        mask = self.maskTrans(mask.convert('RGB'))
        threshhold = 0.5
        ones = mask >= threshhold
        zeros = mask < threshhold

        mask.masked_fill_(ones, 1.0)
        mask.masked_fill_(zeros, 0.0)
        mask = 1 - mask
        inputImage = groundTruth * mask
        inputImage = torch.cat((inputImage, mask[0].view(1, self.cropSize[0], self.cropSize[1])), 0)

        return inputImage, groundTruth, mask
    
    def __len__(self):
        return len(self.imageFiles)
