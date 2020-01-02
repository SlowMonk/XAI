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


datapath = "../datab/file.hdf5"
print("saliency map open......")
#hf = h5py.File(f'../datab/file.hdf5','r')
#sal_maps = np.array(hf['saliencys'])
#print(sal_maps.shape)

#dataloader = utils.load_data_cifar10(batch_size=1,test=False)


'''

for idx, (img, target) in enumerate(dataloader):
    try:
        with h5py.File(datapath, 'a') as hf:
            #sal_maps = np.array(hf['saliencys'])
            dset = hf['saliencys']
            print('appending...{}'.format(sal_maps.shape))
    except:
        print('writing...')
        with h5py.File(datapath, 'w') as hf:
            hf.create_dataset('saliencys', data=img,dtype=np.float32,chunks=(1,)+dataloader.dataset.data.shape[1:])


#hdf = HDF5Store(datapath, dataloader,dataloader.dataset.data.shape[1:])
'''