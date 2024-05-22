import torch
import torch.nn as nn


class Swish(nn.Module):
    def forward(self, x):
        return x*torch.sigmoid(x)
    
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3)
        self.activation1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3)
        self.activation2 = nn.ReLU()
        
        self.layer = nn.Sequential(self.conv1, self.activation1, 
                                   self.conv2, self.activation2)
        
    def forward(self, x):
        return self.layer(x)
    
    
x = torch.rand((1,3,64,64))
net = DoubleConv()
y = net(x)

print(y.shape)