import torch 
import torch.nn as nn

class BatchNorm2d(nn.Module):
    def __init__(self,num_features,eps = 1e-6,momentum = 0.1,affine = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine    #affine tells whether to learn weights(gamma) and bias(beta)
        
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_buffer('weight',None)
            self.register_buffer('bias',None)
        
        self.register_buffer('running_mean',torch.zeros(num_features))
        self.register_buffer('running_var',torch.ones(num_features))
        self.training = True

    def forward(self,x):
        if x.dim() != 4:
            raise ValueError("Batchnorm2d expects 4D input")        
        
        N,C,H,W = x.shape
        if self.training:
            x_peram = x.permute(1,0,2,3).contiguous().view(C,-1)  #.premute change the shape, .contuguoud() record a copy of it in memory
            mean = x_peram.mean(dim = 1)
            var = x_peram.var(dim=1,unbiased=False)
            
            #updated running stats - uses EMA to retain max of old stat and add minimal of new with-
            #.detach() which avoids gradient flowing into running stats
            self.running_mean = (1 - self.momentum)*self.running_mean + self.momentum*mean.detach()  
            self.running_var = (1 - self.momentum)*self.running_var + self.momentum*var.detach()
        
        #training: off , inference :on , so mean and var get default stored values and not updated ones
        else:
            mean = self.running_mean
            var = self.running_var
        
        mean_b = mean.view(1,C,1,1)
        var_b = var.view(1,C,1,1)
        x_norm = (x - mean_b)/torch.sqrt(var_b + self.eps)

        if self.affine:
            w = self.weight.view(1,C,1,1)
            b = self.bias.view(1,C,1,1)

            out = x_norm*w + b
        else:
            out = x_norm
        
        return out
    

