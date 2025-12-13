import torch
from core.model.simple_cnn import SimpleCNN

if '__main__' == __name__:
    model = SimpleCNN()
    x = torch.randn(4,3,32,32,requires_grad=True)
    out = model(x)
    print("output shape",out.shape)

    out.mean().backward()
    print("Backprop OK")