"""
utils
"""
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
])


def load_data_stl10(batch_size=64, test=False):
    if not test:
        train_dset = dsets.STL10(root='./data', split='train', transform=transform, download=True)
    else:
        train_dset = dsets.STL10(root='./data', split='test', transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    print("LOAD DATA, %d" % (len(train_loader)))
    return train_loader

def load_data_cifar10(batch_size=64,test=False):
    if not test:
        train_dset = torchvision.datasets.CIFAR10(root='/mnt/3CE35B99003D727B/input/pytorch/data', train=True,
                                                download=True, transform=transform)
    else:
        train_dset = torchvision.datasets.CIFAR10(root='/mnt/3CE35B99003D727B/input/pytorch/data', train=False,
                                               download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=1, shuffle=True)
    print("LOAD DATA, %d" % (len(train_loader)))
    return train_loader