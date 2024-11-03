# import apex.amp as amp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import os
import sys

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

class Logger(object):
    """
    Write console output to external text file.
    Reference: https://github.com/aamir-mustafa/pcl-adversarial-defense/blob/master/utils.py.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
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


def get_limit(dataset):
    """Attack constraints"""
    if dataset == 'cifar10':
        mu = torch.tensor(cifar10_mean).view(3, 1, 1).to('cuda')
        std = torch.tensor(cifar10_std).view(3, 1, 1).to('cuda')      
    else:
        print('Wrong dataset:', dataset)
        exit()

    upper_limit = ((1 - mu) / std)
    lower_limit = ((0 - mu) / std)

    return std, upper_limit, lower_limit



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


def get_loaders(dir_, batch_size, dataset='cifar10', worker=4, norm=True):
    """Data Loader"""
    if norm:        
        if dataset == 'cifar10':
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
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
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            dataset_normalization = NormalizeByChannelMeanStd(
                mean=cifar10_mean, std=cifar10_std)


    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
            dir_, train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10(
            dir_, train=False, transform=test_transform, download=True)

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
    return train_loader, test_loader, dataset_normalization


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
