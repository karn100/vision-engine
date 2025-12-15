from datasets.cifar10_loader import get_cifar10_dataset

train_ds,val_ds = get_cifar10_dataset()
print(len(train_ds), len(val_ds))