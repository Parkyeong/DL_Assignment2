# import apex.amp as amp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import random_split
import numpy as np
import os
import sys

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)


#Logger 类将控制台输出重定向到文本文件，同时保留输出在控制台上显示。
class Logger(object):
   
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()



def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

def get_loaders(dir_, batch_size, dataset='cifar10', worker=4, norm=True, augs=[],validation=False):
    """Data Loader"""
    
    augmentations = {
        "RandomCrop": transforms.RandomCrop(32, padding=4),
        "RandomHorizontalFlip": transforms.RandomHorizontalFlip(),
        "RandomRotation": transforms.RandomRotation(15),
    }
    transforms_list = []

    for aug in augs:
        transforms_list.append(augmentations[aug])

    if norm:        
        if dataset == 'cifar10':
            train_transform = transforms.Compose([
                *transforms_list,
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(cifar10_mean, cifar10_std),
            ])
        dataset_normalization = None
    else:
        if dataset == 'cifar10':
            train_transform = transforms.Compose([
                *transforms_list,
                transforms.ToTensor(),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            dataset_normalization = NormalizeByChannelMeanStd(
                mean=cifar10_mean, std=cifar10_std
            )

    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            dir_, train=True, transform=train_transform, download=True
        )
        test_dataset = datasets.CIFAR10(
            dir_, train=False, transform=test_transform, download=True
        )

    if validation:
        # Split the training data into 80% train and 20% validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=worker,
        )

    else:
        val_loader = None

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=worker,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=worker,
    )

    if validation:
        return train_loader, val_loader, dataset_normalization
    else:
        return train_loader, test_loader, dataset_normalization
    

#在测试集上评估模型的准确率和损失
def evaluate_standard(test_loader, model):
    """Evaluate without randomization on clean images"""
    test_loss = 0
    test_acc = 0
    n = 0
    model.eval()

    with torch.no_grad():
        for i, (X, y) in enumerate(test_loader):
            X, y = X.to('cuda'), y.to('cuda')
            output = model(X)
            loss = F.cross_entropy(output, y)
            test_loss += loss.item() * y.size(0)
            test_acc += (output.max(1)[1] == y).sum().item()
            n += y.size(0)
    return test_loss/n, test_acc/n
