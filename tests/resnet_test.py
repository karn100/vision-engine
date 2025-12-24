from core.model.resnet.resnet import Resnet18
from core.model.resnet.blocks import BasicBlock
import torch

def resnet18(num_classes = 10):
    return Resnet18(BasicBlock,[2,2,2,2],num_classes)

if "__main__" == __name__:
    model = resnet18()
    x = torch.randn(4,3,32,32)
    y = model(x)
    print(y.shape)