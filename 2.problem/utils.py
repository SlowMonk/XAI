"""
utils
"""
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2

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

def load_data_cifar10(batch_size=128,test=False):
    if not test:
        train_dset = torchvision.datasets.CIFAR10(root='/mnt/3CE35B99003D727B/input/pytorch/data', train=True,
                                                download=True, transform=transform)
    else:
        train_dset = torchvision.datasets.CIFAR10(root='/mnt/3CE35B99003D727B/input/pytorch/data', train=False,
                                               download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    print("LOAD DATA, %d" % (len(train_loader)))
    return train_loader


def rescale_image(images):
    '''
    MinMax scaling

    Args:
        images : images (batch_size, C, H, W)
    '''
    mins = np.min(images, axis=(1,2,3)) # (batch_size, 1)
    mins = mins.reshape(mins.shape + (1,1,1,)) # (batch_size, 1, 1, 1)
    maxs = np.max(images, axis=(1,2,3))
    maxs = maxs.reshape(maxs.shape + (1,1,1,))

    images = (images - mins)/(maxs - mins)
    images = images.transpose(0,2,3,1)

    return images


   # resize to input image size
def resize_image(cam, origin_image):
    original_cam =cam
    original_image= origin_image
    img = np.uint8(Image.fromarray(cam).resize((origin_image.shape[:2]), Image.ANTIALIAS)) / 255
    img = np.expand_dims(cam,axis=2)
    return img