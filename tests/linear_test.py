from core.layers.linear import Linear
import torch

if __name__ == '__main__':
    fc = Linear(512,10)
    x = torch.randn(4,512,requires_grad=True)
    y = fc(x)
    print("Linear out shape",y.shape)
    y.sum().backward()
    print("Grad ok",x.grad is not None)