'''
Training ACGAN
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
parser.add_argument('-save_model', '--save_model', type=str, help='The path to save trained models.',default = 'save_model/ACGAN/')
parser.add_argument('-save_img', '--save_img', type=str, help='The path to save output imgs.',default = 'images/')
parser.add_argument('-utils', '--utils', type=str, default='./utils', help='The path to save trained models.')
args = parser.parse_args()

# import from utils folder
sys.path.insert(1, args.utils)
from utils import CelebA
from torch.utils.data import DataLoader


#from ACGAN import Generator, Discriminator, weights_init
# custom weights initialization called on netG and netD
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
            nn.ConvTranspose2d( nz + 1, ngf * 8, 4, 1, 0, bias=False),  # + 1 for class
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
            
        )
        self.output = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            
        )

    def forward(self, img):
        hidden = self.dis(img)
        output = self.output(hidden)
        classes = self.classifier(hidden)
        return output, classes


nc = 3
nz = 100
ngf = 64
ndf = 64
ngpu = 1 
lr = 0.0001 
beta1 = 0.5 
img_size = 64 
batch_size = 128 
num_epochs = 100 

# make output folder
os.makedirs(args.save_model, exist_ok=True)
os.makedirs(args.save_img, exist_ok=True)

# load the trainset
trainset = CelebA(image_dir='hw3_data/face/train', csv_dir='hw3_data/face/train.csv', 
                  transform=transforms.Compose([
                      transforms.Resize(img_size),
                      transforms.CenterCrop(img_size),
                      transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                  ]))

print('# images in trainset:', len(trainset)) # 

# Use the torch dataloader to iterate through the dataset
dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

images, labels = next(iter(dataloader))
print('Image tensor in each batch:', images.shape, images.dtype)
print('Label tensor in each batch:', labels.shape, labels.dtype)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.savefig(os.path.join(args.save_img, 'training_images.png'))

# Loss function
criterion = torch.nn.BCELoss()

# Initialize generator and discriminator
netG = Generator(nc, nz, ngf, ndf, ngpu).to(device)
netD = Discriminator(nc, nz, ngf, ndf, ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
    netD = nn.DataParallel(netD, list(range(ngpu)))
    
# Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2
netG.apply(weights_init)
netD.apply(weights_init)

# Print the model
print(netG)
print(netD)

# set random seed
manualSeed = 777
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)

# Create batch of latent vectors that we will use to visualize the progression of the generator
fixed_noise = torch.randn(10, nz, 1, 1, device=device)
fixed_noise_1 = torch.cat((fixed_noise, torch.zeros((10, 1, 1, 1), device=device)), dim=1)
fixed_noise_2 = torch.cat((fixed_noise, torch.ones((10, 1, 1, 1), device=device)), dim=1)
fixed_noise = torch.cat((fixed_noise_1, fixed_noise_2), dim=0)  
print("Size of fixed noise: ", fixed_noise.size())  # (20, nz+1, 1, 1)

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output, output_cls = netD(real_cpu)   #.view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output.view(-1), label) + criterion(output_cls.view(-1), data[1].to(device))
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        #print('data[1]:', data[1].shape)  #[128]
        noise = torch.randn(b_size, nz, 1, 1, device=device)  # +1 for class
        noise = torch.cat((noise, data[1].view(b_size, 1, 1, 1).to(device)), dim=1)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output, output_cls = netD(fake.detach())   #.view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output.view(-1), label) + criterion(output_cls.view(-1), data[1].to(device))
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output, output_cls = netD(fake)  # .view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output.view(-1), label) + criterion(output_cls.view(-1), data[1].to(device))
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, nrow=10, padding=2, normalize=True))

        iters += 1
    # save model
    if (epoch+1) % 5 == 0:
        state = {
            'model': 'ACGAN',
            'epoch': epoch,
            'state_dict': [netG.state_dict(), netD.state_dict()],
            'optimizer': [optimizerG.state_dict(), optimizerD.state_dict()],
            'loss': {"d_loss": D_losses, "g_loss": G_losses}
        }
        filename = os.path.join(args.save_model,"epoch{}_checkpoint.pth.tar".format(epoch+1))
        torch.save(state, f=filename)
        
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(os.path.join(args.save_img, 'loss.png'))

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
plt.savefig(os.path.join(args.save_img, 'real_images.png'))
# Plot the fake images from the last epoch
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.savefig(os.path.join(args.save_img, 'fake_images.png'))
