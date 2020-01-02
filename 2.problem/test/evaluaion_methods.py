import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import h5py
from tqdm import tqdm
import torchvision.transforms as transforms
import cv2
from PIL import Image
from collections import OrderedDict
from functools import partial
from torch.nn import functional as F


def adjust_image(dataloader,saliency_map, ratio, eval_method):
    pass
    print("=================================adjust_image=================================")
    data = dataloader.dataset.data
    img = saliency_map
    img_size = data.shape[1:] # cifar10,(3,128,128)
    nb_pixel = np.prod(img_size)
    threshold = int(nb_pixel  * (1-ratio))

    #rank indice
    re_sal_maps = img.reshape(img.shape[0],-1)
    indice = re_sal_maps.argsort().argsort()

    if eval_method == "ROAR":
        mask = indice < threshold
    elif eval_method =="KAR":
        mask = indice >= threshold

    # remove
    print("mask.shape->{}".format(mask.shape))
    print("data.shape->>{}".format(data.shape))
    mask = mask.reshape(data.shape)
    # print("data_shape->{} , mask_shape->{}".format(data.shape,mask.shape))
    # print('dataloader data->{}'.format(data))
    # maskedDataloader = (data * mask).reshape(data.shape)
    # return maskedDataloader
