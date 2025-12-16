import numpy as np
from torchvision.datasets import CIFAR10
import albumentations as A
from torch.utils.data import Dataset
from .transform import get_train_transform,get_val_transform

class CIFAR10Albumentations(Dataset):
    def __init__(self,root,train=True,transform=None,download=True):
        self.dataset = CIFAR10(root=root,train=train,download=download)
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image,label = self.dataset[idx]
        image = np.array(image)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image,label

def get_cifar10_dataset(config):
    train_transform = get_train_transform(config)
    val_transform = get_val_transform()

    train_dataset = CIFAR10Albumentations(
        root=config["data"]["root"],
        train=True,
        transform=train_transform,
        download=True
    )
    val_dataset = CIFAR10Albumentations(
        root=config["data"]["root"],
        train=False,
        transform=val_transform,
        download=True
    )
    return train_dataset,val_dataset