import torch 
import torch.nn as nn

class RelU(nn.Module):
    def __init__(self, inplace = False):
        super().__init__()
        self.inplace = inplace
    
    def forward(self,x):
        return torch.relu(x) if not self.inplace else torch.relu_(x)

class LeakyReLU(nn.Module):
    def __init__(self, negative_slop = 0.01,inplace = False):
        super().__init__()
        self.negatice_slop = negative_slop
        self.inplace = inplace
    
    def forward(self,x):
        return torch.where(x>=0, x, x*self.negatice_slop)
    