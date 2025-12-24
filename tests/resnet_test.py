# from core.model.resnet.resnet import Resnet18
# from core.model.resnet.blocks import BasicBlock
# import torch

# def resnet18(num_classes = 10):
#     return Resnet18(BasicBlock,[2,2,2,2],num_classes)

# if "__main__" == __name__:
#     model = resnet18()
#     x = torch.randn(4,3,32,32)
#     y = model(x)
#     print(y.shape)
#     y.mean().backward()
#     print("Backward OK")

import torch
from core.model.model_builder import build_model
import yaml

def load_config(path="config/resnet.yaml"):
    with open(path,"r") as f:
        config = yaml.safe_load(f)
    return config

cfg = load_config()
model = build_model(cfg)

x = torch.randn(4, 3, 32, 32)
y = model(x)

print("Output shape:", y.shape)