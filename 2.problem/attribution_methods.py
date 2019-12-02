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


class CAMS(object):
    def __init__(self, net,device):
        self.net = net
        self.feature_blobs = []
        self.finalconv_name = 'conv'
        self.net.eval()
        # self.dataloader = dataloader
        self.params = list(self.net.parameters())
        self.weight_softmax = np.squeeze(self.params[-2].cpu().data.numpy())
        self.device = device

    feature_blobs = []

    def hook_feature(self,module, input, output):
        self.feature_blobs.append(output.cpu().data.numpy())

    def returnCAM(self, feature_conv, weight_softmax, class_idx):
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

    def generate_image(self, dataloader):
        i = 0
        for img, target in dataloader:
            print('img_shape->{} target_shape->{} img_shape[0]->{}'.format(img.shape ,target.shape, img[0].shape))
            image_PIL = transforms.ToPILImage()(img[0])
            image_PIL.save('result/test.jpg')
            print("save file")

            self.net._modules.get(self.finalconv_name).register_forward_hook(self.hook_feature)

            img_tensor = img.to(self.device)
            logit , _ = self.net(img_tensor)
            h_x = F.softmax(logit, dim=1).data.squeeze()
            probs, idx = h_x.sort(0, True)
            #print("idx->{} feature_blobs->{}".format(idx,self.feature_blobs))
            output_cam = self.returnCAM(self.feature_blobs[0],self.weight_softmax,[idx[0].item()])
            img = cv2.imread('result/test.jpg')
            height, width, _ = img.shape
            heatmap = cv2.applyColorMap(cv2.resize(output_cam[0], (width, height)), cv2.COLORMAP_JET)
            result = heatmap * 0.3 + img * 0.5
            cv2.imwrite('result/CAM.jpg', result)
            print("generate image")

            i +=1
            if i == 2: break
