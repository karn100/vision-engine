import torch
import torch.nn as nn
from core.layers.conv import Conv2d
from core.layers.batchnorm import BatchNorm2d
from core.layers.linear import Linear
from core.layers.pooling import MaxPool2d
from core.layers.activation import RelU

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            Conv2d(3,32,kernel_size=3,padding=1),
            BatchNorm2d(32),
            RelU(),
            Conv2d(32,32,kernel_size=3,padding=1),
            BatchNorm2d(32),
            RelU(),
            MaxPool2d(2)
        )

        self.conv_block2 = nn.Sequential(
            Conv2d(32,64,kernel_size=3,padding=1),
            BatchNorm2d(64),
            RelU(),
            MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            Linear(64*8*8,128),
            RelU(),
            Linear(128,num_classes)
        )
    
    def forward(self,x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
    