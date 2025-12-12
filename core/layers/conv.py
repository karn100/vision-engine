import torch
import torch.nn as nn

def im2col(x,kernel_size=3,stride=1,padding=0,dilation=1):
    N,C,H,W = x.shape
    KH,KW = kernel_size
    x_p = torch.nn.functional.pad(x,(padding,padding,padding,padding))
    H_p,W_p = x_p.shape[2],x_p.shape[3]

    H_out = (H + 2*padding - dilation*(KH-1) - 1)//stride + 1
    W_out = (W + 2*padding - dilation*(KW-1) - 1)//stride + 1

    cols = x_p.unfold(2,KH,stride).unfold(3,KW,stride)
    cols = cols.contiguous().view(N,C*KH*KW,H_out*W_out)
    return cols

class Conv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride = 1,padding = 0,dilation = 1,bias = True):
        super().__init__()

        if isinstance(kernel_size,int):
            kernel_size = (kernel_size,kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = nn.Parameter(torch.randn(out_channels,in_channels,kernel_size[0],kernel_size[1]))

        if bias == True:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None
            
        nn.init.kaiming_normal_(self.weight,mode='fan_out',nonlinearity='relu')
        
    def forward(self,x):

        N,C_in,H,W = x.shape
        KH,KW = self.kernel_size

        cols = im2col(x,self.kernel_size,self.stride,self.padding,self.dilation)
        w = self.weight.view(self.out_channels,-1)  #.view(a,-1) actually do this that it keeps a unaffected at 1st position and take the remaining things as product for 2nd position. it flattens the dimensions.

        # out = w @ cols
        out = torch.matmul(w,cols)

        if self.bias is not None:
            out = out + self.bias.view(1,-1,1)
        
        H_out = (H + 2*self.padding - self.dilation*(KH - 1) - 1)//self.stride + 1
        W_out = (W + 2*self.padding - self.dilation*(KW - 1) - 1)//self.stride + 1

        out = out.view(N,self.out_channels,H_out,W_out)
        return out