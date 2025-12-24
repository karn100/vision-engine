import torch
import torch.nn as nn
from core.layers.conv import Conv2d
from core.layers.batchnorm import BatchNorm2d
from core.layers.activation import RelU
from core.layers.pooling import AvgPool2d
from core.layers.linear import Linear
from core.model.resnet.blocks import BasicBlock

class Resnet18(nn.Module):
    def __init__(self,block,layers,num_classes=10):
        super().__init__()

        self.in_channels = 64

        self.conv1 = Conv2d(3,64,kernel_size=3,padding=1,stride=1)
        self.bn1 = BatchNorm2d(64)
        self.relu = RelU()

        self.layer1 = self._make_layer_(block,64,layers[0],stride=1)
        self.layer2 = self._make_layer_(block,128,layers[1],stride=2)
        self.layer3 = self._make_layer_(block,256,layers[2],stride=2)
        self.layer4 = self._make_layer_(block,512,layers[3],stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = Linear(512*block.expansion,num_classes)

    def _make_layer_(self,block,out_channels,blocks,stride):
        layers = []

        #This is the main block which is F(x) -> residual block
        layers.append(block(self.in_channels,out_channels,stride))
        self.in_channels = out_channels*block.expansion

        #This is for identity shortcut because stride == 1
        for _ in range(1,blocks):
            layers.append(block(self.in_channels,out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self,x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x

