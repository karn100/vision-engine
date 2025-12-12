import torch
from core.layers.batchnorm import BatchNorm2d

bn = BatchNorm2d(8)
x = torch.randn(4,8,32,32,requires_grad=True)
y = bn(x)

print("bn out shape",y.shape)
y.mean().backward()
print("Grad is oK",x.grad is not None)