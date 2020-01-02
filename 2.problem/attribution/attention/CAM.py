from imports import *
import matplotlib.pyplot as plt
import torch

class CAM(object):
    def __init__(self, net,trainloader,testloader,device,methods):
        print('CAM')
        self.net = net
        self.feature_blobs = []
        self.finalconv_name = 'conv'
        self.net.eval()
        self.params = list(self.net.parameters())
        self.weight_softmax = np.squeeze(self.params[-2].cpu().data.numpy())
        self.device = device
        self.datapath = "../datab/file_mnist.hdf5"
        self.i = 0
        self.dset = ""
        self.methods = methods
        self.masks=[]
        self.trainloader = trainloader
        self.testloader = testloader
        self.final_conv = 'conv'
        self.net._modules.get(self.final_conv).register_forward_hook(self.hook_feature)
        self.dataname = 'CAM_'+ methods + '.h5py'
        self.sal_maps_hf=''
        self.path = '/home/jake/Gits/AI college/XAI/2.problem/'
        #self.saliency_maps  = ''
    #feature_blobs = []
    def hook_feature(self,module, input, output):
        self.feature_blobs.append(output.cpu().data.numpy())


    def returnCAM(self,feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 256x256
        size_upsample = (28, 28)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        for idx in class_idx:
            cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample))
        return output_cam

        #return output_cam
        return cv2.resize(cam_img, size_upsample)

    def get_cam(self,img):

        # print('get_cam')
        self.net.zero_grad()
        x = img[0, :, :].cpu().data.numpy().squeeze()

        # print(img.shape,x.shape)
        params = list(self.net.parameters())
        weight_softmax = np.squeeze(params[-2].data.cpu().numpy())

        img_tensor = img
        img_variable = img_tensor.cuda()
        logit, _ = self.net(img_variable)

        h_x = F.softmax(logit, dim=1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        #print(self.feature_blobs[0])
        #print('idx->',idx)
        CAMs = self.returnCAM(self.feature_blobs[0], weight_softmax, [idx[0].item()])
        #transToPil = transforms.ToPILImage()
        #transToTensor = transforms.ToTensor()
        height, width = img.shape[2:]

        CAM = cv2.resize(CAMs[0], (width, height))
        heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
        result = (x * CAM)
        return CAM, result
    def get_mask(self):
        self.net._modules.get(self.final_conv).register_forward_hook(self.hook_feature)
        for idx,(img,target) in enumerate(self.trainloader):
            CAM,result = self.get_cam(img)
            self.masks.append(CAM)
            if idx%10000 ==0: print(idx,'/',self.trainloader.dataset.data.shape[0],'%')

    def main(self):
        self.get_mask()

    def show_images(self, limit,ratio):
        plt.rcParams['figure.figsize'] = [20, 10]
        for idxr, r in enumerate(ratio):
            print('Ratio-[{}%] removed'.format(r))
            for idx, (img, target) in enumerate(self.trainloader):
                # print(k,idx)
                self.adjust_image(r)
                plt.subplot(1, len(ratio), idx + 1)
                img = img[0, :, :].numpy().squeeze()
                plt.imshow(img)
                if idx == limit: break
            plt.show()


    def save_saliency_maps(self):
        if len(self.masks) ==0:
            self.get_mask()
        #print('./datab/'+self.dataname,len(self.masks))
        paths = '/home/jake/Gits/AI college/XAI/2.problem/attribution/attention/datab/'+self.dataname
        #print('path->',paths)
        with h5py.File(paths, 'w') as hf:
            hf.create_dataset('saliencys', data=self.masks)
    def get_saliency_maps(self):
        #print('get_saliency_maps')
        try:
            #print(self.path +'datab/'+self.datapath)
            hf=h5py.File(f'/home/jake/Gits/AI college/XAI/2.problem/attribution/attention/datab/'+self.dataname,'r')
            #print('hf->',np.array(hf['saliencys']))
            self.sal_maps_hf= np.array(hf['saliencys'])
        except:
            #print('no_file')
            self.save_saliency_maps()

    def adjust_image(self,ratio):
        #print('mask',self.masks)
        self.get_saliency_maps()

        data = self.trainloader.dataset.data
        #print('saliency_maps->',self.sal_maps_hf)
        #print('data-->',data)
        self.trainloader.dataset.data  = torch.tensor(self.sal_maps_hf) * data
        data = self.trainloader.dataset.data
        img_size= data.shape[1:]
        #print(data.shape[1:])
        nb_pixel = np.prod(img_size)
        #print('nb_pixel->{}'.format(nb_pixel))
        threshold= int(nb_pixel * (1-ratio))
        #print('threshold->{}'.format(threshold))

        re_sal_maps = self.sal_maps_hf.reshape(self.sal_maps_hf.shape[0],-1)
        #print('re_sal_maps->',re_sal_maps.shape)
        indice = re_sal_maps.argsort().argsort()
        #print()
        #print('indice->',indice)
        mask = indice < threshold
        data = self.trainloader.dataset.data
        mask = mask.reshape(data.shape)
        #print(mask.shape,data.shape )
        self.trainloader.dataset.data = data * torch.tensor(mask)
        #print( self.trainloader.dataset.data.shape)
        return self.trainloader

    def generate_image(self, img, i):
        # print("=================generate image=============================")
        img = Variable(img, requires_grad=True)

        self.net.modules.get(self.finalconv_name).register_forward_hook(self.hook_feature)

        #print('feature_blobs->{} img.shape->{}'.format(self.feature_blobs,img.shape))

        img_tensor = img.to(self.device)
        logit, _ = self.net(img_tensor)
        h_x = F.softmax(logit, dim=1).data.squeeze()

        weight_softmaxtemp = self.weight_softmax
        feature_blobstemp = self.feature_blobs[0]
        probs, idx = h_x.sort(0, True)
        idx_temp = [idx[0]]
        output_cam = self.returnCAM(self.feature_blobs[0], self.weight_softmax, [idx[0].item()])
        height, width = 28, 28
        heatmap = cv2.applyColorMap(cv2.resize(output_cam[0], (width, height)), cv2.COLORMAP_JET)
        heatmap2 = cv2.resize(output_cam[0], (width, height))

        img = img.detach().numpy()
        img2 = img[0]
        img2 = np.transpose(img2, axes=(1, 2, 0))
        img2 = cv2.resize(img2, (28, 28))
        img2 = cv2.applyColorMap(cv2.resize(output_cam[0], (width, height)), cv2.COLORMAP_JET)

        camsresult = np.array(list(map(resize_image, heatmap, img2)))
        self.feature_blobs = []
        del img_tensor, output_cam, img2, img
        gc.collect()

        # result=zip(camsresult,heatmap,probs.detach().cpu().numpy(), idx.detach().cpu().numpy())

        return camsresult, probs.detach().cpu().numpy(), idx.detach().cpu().numpy()
    def save_saliency_map(self,dataloader,save_dir):
        i = 0
        #print("==============save_saliency_map============")
        dataloadertemp = dataloader
        img_size= dataloader.dataset.data.shape[1:]
        dim = len(img_size)
        if dim ==2:
            img_size = img_size +(3,)

        for idx, (img, target) in enumerate(dataloader):
            idx +=1

            if True:
                try:
                    with h5py.File(self.datapath, 'a') as hf:
                        #print(hf['saliencys'].shape[0])
                        i = hf['saliencys'].shape[0]
                        #print(idx)
                        i_temp = i + 1
                        if i_temp==idx:

                            sal_maps_b, probs_b, preds_b = self.generate_image(img, idx)
                            sal_maps_b = np.transpose(sal_maps_b, axes=(3,0, 1, 2))


                            if i%500==0:
                                print('appending....original saliency shape->{} {}/50000%'.format(hf["saliencys"].shape,i ))
                            dset = hf['saliencys']
                            dset.resize((i + 1,) + img_size)
                            dset[i] = [sal_maps_b]


                            dprobs = hf['probs']
                            dprobs.resize((i+1,) + probs_b.shape)
                            dprobs[i] = [probs_b]

                            dpreds = hf['preds']
                            dpreds.resize((i + 1,) + preds_b.shape)
                            dpreds[i] = [preds_b]

                            i += 1
                            del i,i_temp,sal_maps_b,preds_b,dset,dprobs,dpreds
                            gc.collect()

                            hf.flush()
                            hf.close()
                except Exception as e:
                    print('error',e)

                    sal_maps_b, probs_b, preds_b = self.generate_image(img, 0)
                    #with  self.generate_image(img, idx) as result:
                    #    sal_maps_b, heat_map_b, probs_b, preds_b = result
                    sal_maps_b = np.transpose(sal_maps_b, axes=(3, 0, 1, 2))

                    print("writing......".format(sal_maps_b.shape))
                    print(sal_maps_b.shape,img_size,probs_b.shape,preds_b.shape)
                    probs_b = np.array([], dtype=np.float32).reshape((0,) + probs_b.shape)
                    preds_b = np.array([], dtype=np.uint8).reshape((0,) + preds_b.shape)
                    print(probs_b.shape,preds_b.shape,probs_b.shape[1:],preds_b.shape[1:])
                    with h5py.File(self.datapath, 'w') as hf:
                        hf.create_dataset('saliencys', data=sal_maps_b,maxshape=(None,) + img_size,chunks=(1,) + img_size)
                        hf.create_dataset('probs', data=probs_b,maxshape=(None,) + probs_b.shape[1:],chunks=(1,) + probs_b.shape[1:])
                        hf.create_dataset('preds', data=preds_b,maxshape=(None,) + preds_b.shape[1:],chunks=(1,)+ preds_b.shape[1:])
                        hf.close()
                        i+=1

#@contextmanager
class HDF5Store(object):

    def __init__(self, datapath, dataset, shape, dtype=np.float32, compression="gzip", chunk_len=1):
        self.datapath = datapath
        self.dataset = dataset
        self.shape = shape
        self.i = 0

        with h5py.File(self.datapath, mode='w') as h5f:
            self.dset = h5f.create_dataset(
                dataset,
                shape=(0,) + shape,
                maxshape=(None,) + shape,
                dtype=dtype,
                compression=compression,
                chunks=(chunk_len,) + shape)

    def append(self, values):
        with h5py.File(self.datapath, mode='a') as h5f:
            dset = h5f[self.dataset]
            dset.resize((self.i + 1,) + self.shape)
            dset[self.i] = [values]
            self.i += 1
            h5f.flush()