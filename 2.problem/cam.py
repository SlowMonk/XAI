import os
import torch
from torch.nn import functional as F
import cv2
import numpy as np
import torchvision.transforms as transforms
import utils
import model
from attribution_methods import *
print("===cam===")

if not os.path.exists('./result'):
    os.mkdir('result/')

classes = ('airplance', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey',
        'ship', 'truck')
train_loader = utils.load_data_cifar10(batch_size=1,test=False)
test_loader = utils.load_data_cifar10(batch_size=1, test=True)
is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")
net = model.load_net().to(device)
finalconv_name = 'conv'

cam = CAMS(net,device)
cam.adjust_image(train_loader,0.1)