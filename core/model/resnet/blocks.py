import torch
import torch.nn as nn
from core.layers.conv import Conv2d
from core.layers.batchnorm import BatchNorm2d
from core.layers.pooling import MaxPool2d
from core.layers.activation import RelU

class BasicBlock(nn.Module):
    
    expansion = 1

    def __init__(self,in_channels,out_channels,stride = 1):
        super().__init__()
        self.conv1 = Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=stride,bias=False)
        self.bn1 = BatchNorm2d(out_channels)
        self.relu = RelU()

        self.conv2 = Conv2d(out_channels,out_channels,kernel_size=3,padding=1,stride=1,bias=False)
        self.bn2 = BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                Conv2d(in_channels,out_channels,kernel_size=1,stride=stride,bias=False),
                BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self,x):

        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out
    