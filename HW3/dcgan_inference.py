'''
generate fig1_2.jpg (fake image) from DCGAN
'''
import os
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision.transforms as transforms


parser = argparse.ArgumentParser()
parser.add_argument('-ckpt', '--checkpoint', type=str, help='The path to load trained models.')
parser.add_argument("save_img",  help='The path to save output imgs.')
parser.add_argument('-utils', '--utils', type=str, default='./utils', help='The path to save trained models.')
args = parser.parse_args()

from torch.utils.data import DataLoader


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    
class Generator(nn.Module):
    def __init__(self, nc, nz, ngf, ndf, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.gen = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, z):
        return self.gen(z)

class Discriminator(nn.Module):
    def __init__(self, nc, nz, ngf, ndf, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.dis = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.dis(x)


nc = 3 
nz = 100 
ngf = 64 
ndf = 64 
ngpu = 0 
img_size = 64 
# make output folder
os.makedirs(args.save_img, exist_ok=True)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
#print(device)
# Initialize generator and discriminator
netG = Generator(nc, nz, ngf, ndf, ngpu).to(device)
netD = Discriminator(nc, nz, ngf, ndf, ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
    netD = nn.DataParallel(netD, list(range(ngpu)))
    
# Print the model
print(netG)
print(netD)

# Load the model
checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
netG.load_state_dict(checkpoint['state_dict'][0])
netD.load_state_dict(checkpoint['state_dict'][1])
netG.eval()
netD.eval()

# set random seed
manualSeed = 777
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)

# Create batch of latent vectors that we will use to visualize the progression of the generator
fixed_noise = torch.randn(32, nz, 1, 1, device=device)


# Check how the generator is doing by saving G's output on fixed_noise

fake = netG(fixed_noise).detach().cpu()
    
fake_img = vutils.make_grid(fake, padding=2, normalize=True)

# Plot the fake images from the last epoch
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(fake_img,(1,2,0)))
plt.savefig(os.path.join(args.save_img, 'fig1_2.jpg'))
print("GAN Finished !!")
