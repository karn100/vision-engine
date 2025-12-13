import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

class Trainer:
    def __init__(self,model,train_loader,val_loader,device,config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.criterion = nn.CrossEntropyLoss()

        if config["optimizer"]["type "].lower() == "sgd":
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=config["optimizer"]["lr"],
                weight_decay=config["optimizer"]["weight_decay"]
            )
        elif config["optimizer"]["type"].lower() == "adam":
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=config["optimizer"]["lr"],
                weight_decay=config["optimizer"]["weight_decay"]
            )
        else:
            raise ValueError(f"Unsupported optimizer: {config["optimizer"]["type"]}")
        
        self.scheduler = optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=config["scheduler"]["step_size"],
            gamma=config["scheduler"]["gamma"]
        )
        
        self.save_dir = config["trainer"]["save_dir"]
        os.makedirs(self.save_dir,exist_ok=True)
    
    