import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import random_split

def get_transfrom(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32,padding=1),
            transforms.ToTensor(),
            transforms.Normalize(
                 mean=(0.4914, 0.4822, 0.4465),
                 std=(0.2023, 0.1994, 0.2010)
            )
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                 mean=(0.4914, 0.4822, 0.4465),
                 std=(0.2023, 0.1994, 0.2010)
            )
        ])

def get_cifar10_dataset(root="./data",augment=True,val_split=0.1):
    
    full_train = CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=get_transfrom(train=augment)
    )

    val_transform = get_transfrom(train=False)

    val_size = int(len(full_train)*val_split)
    train_size = len(full_train) - val_size

    train_dataset,val_dataset = random_split(
        full_train,[train_size,val_size]
    )

    val_dataset.dataset.transform = val_transform

    return train_dataset,val_dataset

