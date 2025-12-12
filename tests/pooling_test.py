import torch
from core.layers.pooling import MaxPool2d

if '__main__' == __name__:
    pool = MaxPool2d(2)
    x = torch.randn(1,3,32,32,requires_grad=True)
    y = pool(x)
    print("out shape",y.shape)
    y.sum().backward()
    print("Grad is OK",x.grad is not None)
