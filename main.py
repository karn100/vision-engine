import torch
import yaml
import numpy as np
import random
from torch.utils.data import DataLoader
from datasets.cifar10_loader import get_cifar10_dataset
from core.model.simple_cnn import SimpleCNN
from core.train.trainer import Trainer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(path="config/default.yaml"):
    with open(path,"r") as f:
        config = yaml.safe_load(f)
    return config

def main():
    config = load_config()
    set_seed(config["train"]["seed"])
    device = torch.device(config["train"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    train_dataset,val_dataset = get_cifar10_dataset(config=config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=config["data"]["shuffle"],
        num_workers=config["data"]["num_workers"],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle = False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True
    )

    model = SimpleCNN(num_classes=config["model"]["num_classes"])
    print(f"Model {config['model']['name']} Initialized")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    trainer.fit()

if __name__ == "__main__":
    main()