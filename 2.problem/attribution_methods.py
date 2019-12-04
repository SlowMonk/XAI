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
from utils import *


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
        #print("캠이미지 사이즈:{}".format(cam_img.shape))
        output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

    def generate_image(self, img, i):
        #print("=================generate image=============================")
        #print("img_size->{} dtype->{}".format(img.shape,img.dtype))
        img = Variable(img, requires_grad=True)

        #image_PIL = transforms.ToPILImage()(img[0])
        #image_PIL.save('result/test{}.jpg'.format(i))

        self.net._modules.get(self.finalconv_name).register_forward_hook(self.hook_feature)

        img_tensor = img.to(self.device)
        logit , _ = self.net(img_tensor)
        h_x = F.softmax(logit, dim=1).data.squeeze()


        probs, idx = h_x.sort(0, True)
        output_cam = self.returnCAM(self.feature_blobs[0],self.weight_softmax,[idx[0].item()])
        height, width = 32,32
        heatmap = cv2.applyColorMap(cv2.resize(output_cam[0], (width, height)), cv2.COLORMAP_JET)

        #ndarray change
        #img = cv2.imread('result/test{}.jpg'.format(i))

        img =img.detach().numpy()
        img2 = img[0]
        img2 = np.transpose(img2,axes=(1,2,0))
        img2=cv2.resize(img2,(32,32))
        #img2 =  img[0]
        #np.transpose(img,axes=(2,0,1))


        modify_image = heatmap * 0.3 + img2 * 0.5

        camsresult = np.array(list(map(resize_image, modify_image, img2)))
        return camsresult, probs.detach().cpu().numpy(), idx.detach().cpu().numpy()

    def save_saliency_map(self,dataloader,save_dir):
        dataloadertemp = dataloader
        print("dataloader.shape->{}".format(dataloader.dataset.data.shape))
        img_size= dataloader.dataset.data.shape[1:]
        dim = len(img_size)
        if dim ==2:
            img_size = img_size +(1,)

        sal_maps = np.array([], dtype=np.float32).reshape((0,) + img_size)
        probs = np.array([], dtype=np.float32)
        preds = np.array([], dtype=np.uint8)
        for idx, (img, target) in enumerate(dataloader):

            sal_maps_b, probs_b, preds_b = self.generate_image(img, idx)
            sal_maps_b = np.transpose(sal_maps_b, axes=(3,0, 1, 2))

            sal_maps = np.vstack([sal_maps,sal_maps_b])
            probs = np.append(probs, probs_b)
            preds = np.append(preds,preds_b)
            #if idx ==2: break
            if idx % 100 == 0:
                print("idx->{} sal_maps->{}, probs->{} , preds->{}".format(idx,sal_maps.shape,probs.shape,preds.shape))
        print("result::{}",sal_maps.size)
        name = "file"
        save_dir += "/" + name +  ".hdf5"
        with h5py.File(save_dir, 'w') as hf:
            hf.create_dataset('saliencys', data=sal_maps)
            hf.create_dataset('probs', data=probs)
            hf.create_dataset('preds', data=preds)
            hf.close()
        print('Save_saliency_maps')


