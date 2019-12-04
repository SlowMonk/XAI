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
        #print("캠이미지 사이즈:{}".format(cam_img.shape))
        output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

    def generate_image(self, img, i):
        print("=================generate image=============================")
        print("img_size->{} dtype->{}".format(img.shape,img.dtype))
        img = Variable(img, requires_grad=True)

        #i = 0
        #for img, target in dataloader:
        #print('img_shape->{} target_shape->{} img_shape[0]->{}'.format(img.shape ,target.shape, img[0].shape))
        #image_PIL = transforms.ToPILImage()(img[0])
        #image_PIL.save('result/test{}.jpg'.format(i))
        #print("save file")

        self.net._modules.get(self.finalconv_name).register_forward_hook(self.hook_feature)

        img_tensor = img.to(self.device)
        logit , _ = self.net(img_tensor)
        h_x = F.softmax(logit, dim=1).data.squeeze()

        probs, idx = h_x.sort(0, True)
        #print('preds란->{} and idx란->{}'.format(probs, idx))

        #probs, idx = h_x.max(1)

        #probs = probs.max(1)
        #idx = probs.max(1)
        #print(img.shape[-2:])
        output_cam = self.returnCAM(self.feature_blobs[0],self.weight_softmax,[idx[0].item()])
        #print('11:output_cam->{}',output_cam.shape)
        #img = cv2.imread('result/test{}.jpg'.format(i))
        print("iamge_shape->{}".format(img.shape))

        #img = cv2.resize(img,(32,32))

        #height, width, _ = img.shape
        #_,_,height, width = img.shape
        height, width = 32,32
        heatmap = cv2.applyColorMap(cv2.resize(output_cam[0], (width, height)), cv2.COLORMAP_JET)

        #output_cam = cv2.resize(output_cam[0], (width,height))
        print("13.original image_size->",torch.FloatTensor(img[0]).shape)
        #print("14.outputcam->{}".format(output_cam.shape))
        #img = cv2.resize(img,(32,32))
        #img=np.transpose(img,axes=(2,0,1))


        print("9: heatmap->{}  img->{}".format(heatmap.shape,img.shape))
        result = heatmap * 0.3 + img * 0.5

        #cv2.imwrite('result/CAM{}.jpg'.format(i), result)
        #print("1.generate image -> result shape==", result.shape)

        #img=np.transpose(img,axes=(2,0,1))
        #print('2.oridingal image->{}'.format(img.shape))

        #print("3.original 이미지 사이즈:{}".format(result.shape[-2:]))
        # resize to input image size
        def resize_image(gradcam, origin_image):
            #print("imagefrom_array->{}".format(Image.fromarray(gradcam)))
            #img = np.uint8(Image.fromarray(gradcam).resize(((32,32)), Image.ANTIALIAS)) / 255
            dim = (32,32)
            #img = cv2.resize(gradcam, dim, interpolation=cv2.INTER_AREA) /255
            img =  cv2.resize(gradcam, dim)
            #print('img_size->', img.shape)

            #print('imge_shape_>{}'.format(img.shape))
            #img = np.expand_dims(img, axis=2)
            #print('imge_shape2_>{}'.format(img.shape))


            return img
        cams = result
        cams = np.array(list(map(resize_image, result, img)))
        #print("4.리사이즈 이미지->{}".format(cams.shape))

        #i +=1
        #if i == 10: break

        return cams , probs.detach().cpu().numpy(), idx.detach().cpu().numpy()

    def save_saliency_map(self,dataloader,save_dir):
        img_size= dataloader.dataset.data.shape[1:]
        dim = len(img_size)
        if dim ==2:
            img_size = img_size +(1,)

        sal_maps = np.array([], dtype=np.float32).reshape((0,) + img_size)
        probs = np.array([], dtype=np.float32)
        preds = np.array([], dtype=np.uint8)
        #print("9.sal_maps->{} , probs->{}, preds->{}".format(sal_maps.shape,probs.shape,preds.shape))

        for idx, (img, target) in enumerate(dataloader):

            sal_maps_b, probs_b, preds_b = self.generate_image(img, idx)
            sal_maps_b = np.transpose(sal_maps_b, axes=(1, 2, 0))

            #print('6.sam_maps->{} sal_maps_b->{}'.format(sal_maps.shape,sal_maps_b.shape))
            sal_maps = np.append(sal_maps,sal_maps_b)
            #print('7.sal_maps->{}'.format(sal_maps.shape))
            probs = np.append(probs, probs_b)
            preds = np.append(preds,preds_b)
            if idx ==2: break
            if idx % 100 == 0:
                print("idx->{} sal_maps->{}".format(idx,sal_maps.shape))
                # print("4.dataloader 사이즈{}".format(dataloader.dataset.data.shape))
                # print("=================adjust image=============================")
                # print("5.idx->{} img_size->{}, target->{}".format(idx, img.shape, target))
                # print("================================================================")
        print("result::{}",sal_maps.size)
        name = "file"
        save_dir += "/" + name +  ".hdf5"
        with h5py.File(save_dir, 'w') as hf:
            hf.create_dataset('saliencys', data=sal_maps)
            hf.create_dataset('probs', data=probs)
            hf.create_dataset('preds', data=preds)
            hf.close()
        print('Save_saliency_maps')


