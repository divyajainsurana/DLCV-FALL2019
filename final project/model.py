import torch.nn as nn
import torch
import torchvision.models

class Base_Network(nn.Module):
    def __init__(self):
        super(Base_Network, self).__init__()

        resnet18 = torchvision.models.resnet34(pretrained=True)
        
        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.l1 = resnet18.layer1
        self.l2 = resnet18.layer2
        self.l3 = resnet18.layer3
        self.l4 = resnet18.layer4
        # Decoder:
        
        self.transconv1 = nn.Sequential(
				nn.ConvTranspose2d(512,256,kernel_size=4,stride =2,padding =1,bias=False),
       				nn.LeakyReLU(0.2, inplace=True),nn.Dropout(p=0.25),
			  )

        self.transconv2 = nn.Sequential(
				nn.ConvTranspose2d(512,128,kernel_size=4,stride =2,padding =1,bias=False),
        			nn.LeakyReLU(0.2, inplace=True),nn.Dropout(p=0.25),
        		  )
        self.transconv3 = nn.Sequential(
				nn.ConvTranspose2d(256,64,kernel_size=4,stride =2,padding =1,bias=False),
        			nn.LeakyReLU(0.2, inplace=True),nn.Dropout(p=0.25),
			  )
        self.transconv4 = nn.Sequential(
				nn.ConvTranspose2d(128,64,kernel_size=3,stride =1,padding =1,bias=False),
        			nn.LeakyReLU(0.2, inplace=True),nn.Dropout(p=0.25),
			  )
        self.transconv5 = nn.Sequential(
				nn.ConvTranspose2d(128,16,kernel_size=4,stride =2,padding =1,bias=False),
        			nn.LeakyReLU(0.2, inplace=True),nn.Dropout(p=0.25),
			 )
        self.conv2 = nn.Sequential(nn.ConvTranspose2d(16,3,kernel_size=4, stride=2, padding=1),
                                nn.LeakyReLU(0.2, inplace=True),nn.Dropout(p=0.25),
                        )
        self.conv3 = nn.Sequential(nn.Conv2d(3,64,4,2,1),nn.ReLU() )
        self.conv3_m = nn.Sequential(nn.Conv2d(1,64,4,2,1),nn.ReLU() )
        self.conv4 = nn.Sequential(nn.Conv2d(64,128,4,2,1),nn.ReLU() )
        self.conv4_m = nn.Sequential(nn.Conv2d(64,128,4,2,1),nn.ReLU() )
        self.conv5 = nn.Sequential(nn.Conv2d(128,256,4,2,1),nn.ReLU())
        self.conv5_m = nn.Sequential(nn.Conv2d(128,256,4,2,1),nn.ReLU())
        self.dc5 = nn.Sequential(nn.ConvTranspose2d(256,128,4,2,1),nn.ReLU())
        self.dc5_m = nn.Sequential(nn.ConvTranspose2d(256,128,4,2,1),nn.ReLU())
        self.dc6 = nn.Sequential(nn.ConvTranspose2d(256,64,4,2,1),nn.ReLU())
        self.dc6_m = nn.Sequential(nn.ConvTranspose2d(256,64,4,2,1),nn.ReLU())
        self.dc7 = nn.Sequential(nn.ConvTranspose2d(128,3,4,2,1))
        self.dc7_m = nn.Sequential(nn.ConvTranspose2d(128,1,4,2,1))
        self.sigmoid = nn.Sigmoid() 
    def forward(self, input, mask):
        out1 = self.conv1(input)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.maxpool(out1)
        out2 = self.l1(out1)
        out3 = self.l2(out2)
        out4 = self.l3(out3)
        out5 = self.l4(out4)
        x = out5
        x = self.transconv1(x)
        x = self.transconv2(torch.cat((x,out4),1))
        #x = self.maxpool(x)
        x = self.transconv3(torch.cat((x,out3),1))
        x = self.transconv4(torch.cat((x,out2),1))
        x = self.transconv5(torch.cat((x,out1),1))
#        print(x.shape)
       # x = self.transconv6(torch.cat((x, out1), 1))
        out6 = self.conv2(x)
        out7_m = self.conv3_m(mask)
        out7= self.conv3(out6)
        out8_m = self.conv4_m(out7_m)
        out8 = self.conv4(out7)
        out9_m = self.conv5(out8_m)
        out9 = self.conv5(out8)
        x_m = self.dc5_m(out9_m)
        x = self.dc5(out9)
        x_m = self.dc6_m(torch.cat((x_m,out8_m),1))
        x = self.dc6(torch.cat((x,out8),1))
        x_m = self.dc7_m(torch.cat((x_m,out7_m),1))
        x = self.dc7(torch.cat((x,out7),1))
        mask = self.sigmoid(x_m)
        x = self.sigmoid(x)
        d = torch.mul(x, 1 - mask)
        s = torch.mul(input, mask)
        x = s + d
    

        return x
