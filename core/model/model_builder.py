from core.model.resnet.blocks import BasicBlock
from core.model.resnet.resnet import Resnet18

def build_model(cfg):
    if cfg["model"]["name"].lower() == "resnet18":
        if cfg["model"]["block"] == "BasicBlock":
            block = BasicBlock
        else:
            raise ValueError("Unsupported Block")
        
        return Resnet18(block=block,
                        layers=cfg["model"]["layers"],
                        num_classes=cfg["model"]["num_classes"])