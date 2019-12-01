import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import h5py
from tqdm import tqdm

import cv2
from PIL import Image
from collections import OrderedDict
from functools import partial

class CAMS(object):
    def __int__(self,model):
        self.model = model
        self.model.eval()
        self.net = model
        self.feature_blobs = []
        self.finalconv_name = 'conv'
        #self.dataloader = dataloader

    def hook_feature(self,module, input, output):
        self.feature_blobs.append(output.cpu().data.numpy())


    def returnCAM(self,feature_conv, weight_softmax, class_idx):
        size_upsample = (128, 128)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

    def generate_image(self,dataloader):
        # print("hook features",output.cpu().data.numpy())
        self.net.modules.get(self.finalconv_name).register_forward_hook(self.hook_feature)

        params = list(self.net.parameters())
        # get weight only from the last layer(linear)
        weight_softmax = np.squeeze(params[-2].cpu().data.numpy())
        i = 0
        for img,target in dataloader:
            print(img.shape,target.shape)

            if i==2:break
