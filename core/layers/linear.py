import torch
import torch.nn as nn
import math
import torch.nn.init as init

class Linear(nn.Module):
    def __init__(self, in_features,out_features,bias = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features,in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias',None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        #kaiming uniform-like initialization is done:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in,_ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1/math.sqrt(fan_in)
            init.uniform_(self.bias,-bound,bound)
        
    def forward(self,x):
        original_shape = x.shape
        
        #now we flattens the x hwich has shape(N,*,in_features) , where final shape becomes (N*'anything here', in_features)
        x_flat = x.view(-1,self.in_features) 

        #This is basic matrix multiplication where x_flat(N,in_features) @ weight.t where t is transpose
        #self.weight(out_features,in_features), where self.weight.t(in_features,out_features)
        out = torch.matmul(x_flat,self.weight.t())
        if self.bias is not None:
            #This reshapes the self.bias from (out_features) to (1,out_features)
            out = out + self.bias.view(1,-1)
        #here it confirms the out dims by removing the last dimension of original_shape which is in_feature
        # and replace it with self.out_feature, so- out dims are(N,out_feature)    
        out = out.view(*original_shape[:-1],self.out_features)
        return out