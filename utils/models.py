import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop


class Swish(nn.Module):
    def forward(self, x):
        return x*torch.sigmoid(x)
    
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.activation1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.activation2 = nn.ReLU()
        
        self.layer = nn.Sequential(
            self.conv1,
            self.norm1,
            self.activation1,
            self.conv2,
            self.norm2,
            self.activation2
        )
        
    def forward(self, x):
        return self.layer(x)
    
    
class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.size = size
        
    def forward(self, x):
        return nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)(nn.Upsample(self.size)(x))
    
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super().__init__()
        
        # Encoder
        self.conv1 = DoubleConv(in_channels=in_channels, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = DoubleConv(in_channels=64, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = DoubleConv(in_channels=128, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.conv4 = DoubleConv(in_channels=256, out_channels=512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.conv5 = DoubleConv(in_channels=512, out_channels=1024)
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        
        # Decoder
        self.upconv1 = UpConv(in_channels=1024, out_channels=512, size=(56,56))
        self.conv6 = DoubleConv(in_channels=1024, out_channels=512)
        
        self.upconv2 = UpConv(in_channels=512, out_channels=256, size=(104,104))
        self.conv7 = DoubleConv(in_channels=512, out_channels=256)
        
        self.upconv3 = UpConv(in_channels=256, out_channels=128, size=(200,200))
        self.conv8 = DoubleConv(in_channels=256, out_channels=128)
        
        self.upconv4 = UpConv(in_channels=128, out_channels=64, size=(392,392))
        self.conv9 = DoubleConv(in_channels=128, out_channels=64)
        
        # Prediction head
        self.head = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool1(x1))
        x3 = self.conv3(self.pool2(x2))
        x4 = self.conv4(self.pool3(x3))
        x5 = self.conv5(self.pool4(x4))
        
        xu5 = self.upconv1(x5)
        x6 = self.conv6(torch.cat((CenterCrop(xu5.shape[2:])(x4), xu5), dim=1))
        xu6 = self.upconv2(x6)
        x7 = self.conv7(torch.cat((CenterCrop(xu6.shape[2:])(x3), xu6), dim=1))
        xu7 = self.upconv3(x7)
        x8 = self.conv8(torch.cat((CenterCrop(xu7.shape[2:])(x2), xu7), dim=1))
        xu8 = self.upconv4(x8)
        x9 = self.conv9(torch.cat((CenterCrop(xu8.shape[2:])(x1), xu8), dim=1))
        
        return nn.Upsample(x.shape[2:])(self.head(x9))
        
        
        
        
        