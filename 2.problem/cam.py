import os
import torch
from torch.nn import functional as F
import cv2
import numpy as np
import torchvision.transforms as transforms
import utils
import model
from attribution_methods import *
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from evaluaion_methods import *
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
save_dir = "./datab"
ratio = 0.1

#save mask image
print("trainloader size->{}".format(train_loader.dataset.data.shape))
cam = CAMS(net,device)
cam.save_saliency_map(train_loader,save_dir)

#adjust image
hf = h5py.File(f'./datab/file.hdf5','r')
cifar10 = h5py.File(f'./datab/cifar10_GB-GC_steps50_ckp5_sample0.1.hdf5','r')

sal_maps = np.array(hf['saliencys'])
#sal_cifar10 = np.array(cifar10['saliencys'])

print('sal_maps->{}'.format(sal_maps.shape))
#print("cifar10->{}".format(cifar10.shape))
#print('cifar10->{}'.format(sal_maps2.shape))

#data_lst = adjust_image(train_loader,sal_maps,ratio,"ROAR")