import torch
import torch.nn as nn
import torch.nn.functional as F
class MaxPool2d(nn.Module):
    def __init__(self, kernel_size = 2,stride = None,padding = 0):
        super().__init__()

        if isinstance(kernel_size,int):
            self.KH = self.KW = kernel_size
        else:
            self.KH,self.KW = kernel_size
        #we set default stride as same as kernel_size
        self.stride = stride if(stride is not None) else kernel_size
        self.padding = padding
    
    def forward(self,x):
        N,C,H,W = x.shape
        x_p = F.pad(x,(self.padding,self.padding,self.padding,self.padding),value=float('inf'))
        KH,KW,S = self.KH,self.KW,self.stride
        patches = x_p.unfold(2,KH,S).unfold(3,KW,S)
        patches = patches.contiguous().view(N,C,patches.shape[2],patches.shape[3],KH*KW)
        out,_ = patches.max(dim=-1)
        return out

class AvgPool2d(nn.Module):
    def __init__(self, kernel_size = 2,stride = None,padding = 0):
        super().__init__()

        if isinstance(kernel_size,int):
            self.KH = self.KW = kernel_size
        else:
            self.KH,self.KW = kernel_size
        self.stride = stride if(stride is not None) else kernel_size
        self.padding = padding
    
    def forward(self,x):
        N,C,H,W = x.shape
        x_p = F.pad(x,(self.padding,self.padding,self.padding,self.padding),value=float('inf'))
        KH,KW,S = self.KH,self.KW,self.stride
        patches = x_p.unfold(KH,2,S).unfold(KW,3,S)
        patches = patches.contiguous().view(N,C,patches.shape[2],patches.shape[3],KH*KW)
        out,_ = patches.mean(dim=-1)
        return out
    