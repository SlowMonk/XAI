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
from imports import *

transform = transforms.Compose([
    #transforms.Resize(32),
    transforms.ToTensor(),
])

transform_mnist = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
])

'''
def load_data_stl10(batch_size=64, test=False):
    if not test:
        train_dset = dsets.STL10(root='./data', split='train', transform=transform, download=True)
    else:
        train_dset = dsets.STL10(root='./data', split='test', transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True)
    print("LOAD DATA, %d" % (len(train_loader)))
    return train_loader
'''
def load_data_cifar10(batch_size=128,test=False):
    if not test:
        train_dset = torchvision.datasets.CIFAR10(root='/mnt/3CE35B99003D727B/input/pytorch/data', train=True,
                                                download=True, transform=transform)
    else:
        train_dset = torchvision.datasets.CIFAR10(root='/mnt/3CE35B99003D727B/input/pytorch/data', train=False,
                                               download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=False)
    print("LOAD DATA, %d" % (len(train_loader)))
    return train_loader

def load_data_mnist(batch_size=128,test=False):
    if not test:
        train_dset = torchvision.datasets.MNIST(root='/mnt/3CE35B99003D727B/input/pytorch/data', train=True,
                                                download=True, transform=transform_mnist)
    else:
        train_dset = torchvision.datasets.MNIST(root='/mnt/3CE35B99003D727B/input/pytorch/data', train=False,
                                               download=True, transform=transform_mnist)
    train_loader = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=False)
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
    #print('cam->{} original_image->{}'.format(cam.shape,origin_image.shape))
    img = np.uint8(Image.fromarray(cam).resize((origin_image.shape[:2]), Image.ANTIALIAS)) / 255
    img = np.expand_dims(cam,axis=2)
    return img




class TRAIN(nn.Module):
    def __init__(self, net,device):
        super(TRAIN, self).__init__()
        self.min_loss = 999
        self.best_accuracy = 0
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer =  torch.optim.Adam(net.parameters(),lr=0.0001)
        self.net = net
        self.device = device
        self.epoch_loss = 0
        self.accarr = []
        self.setparr = []
        self.total = 0
        self.correct = 0
        self.test_loss = 0


    def train(self,epoch,trainloader,step):
        #print("training....")

        for i , (images, targets) in enumerate(trainloader):
            #print(images.shape)
            self.net.train()

            images, targets = images.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            if self.net.name =='CNN_mnist':
                outputs ,_ = self.net(images)
            else:
                 outputs= self.net(images)
            loss = self.criterion(outputs,targets)
            loss.backward()
            self.optimizer.step()

            self.epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            self.total += targets.size(0)
            self.correct += predicted.eq(targets).sum().item()

            if (i + 1) % 10000 == 0:
                print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f  correct: %.4f %%'
                      % (epoch + 1, 10, i + 1, len(trainloader), loss.item(),100.*self.correct/self.total))

        avg_epoch_loss = self.epoch_loss / len(trainloader)

        print("Epoch: %d, Avg Loss: %.4f" % (epoch + 1, avg_epoch_loss))

    def test(self,epoch,testloader,step,source):
        #print("testing....")
        best_accuracy = self.best_accuracy
        self.source = source
        self.net.eval()
        test_loss = 0
        #correct = 0
        #total = 0
        with torch.no_grad():
            for i , (images,target) in enumerate(testloader):
                inputs , targets = images.to(self.device) , target.to(self.device)
                if self.net.name == 'CNN_mnist':
                    outputs, _ = self.net(inputs)
                else:
                    outputs = self.net(inputs)
                self.total += targets.size(0)
                _ ,predicted = outputs.max(1)
                self.correct += predicted.eq(targets).sum().item()
            #save_file(100.*correct/total,step,save_dir,methods,source)
            self.accarr.append(self.correct/self.total)
            self.setparr.append(step/10)
            print("test accuracy:{} %% ".format(100.*self.correct/self.total))
        acc = 100.*self.correct/self.total
        if acc > best_accuracy:
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            state = {
                'net': self.net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            print("saving.............{}_name->{}%".format(acc,'model/{}_adjust_{}_{}.pth'.format(self.source,step,self.net.name)))
            torch.save(state,'./checkpoint/{}_adjust_{}.pth'.format(self.source,step))
            torch.save(self.net.state_dict(), '/home/jake/Gits/AI college/XAI/2.problem/model_weights/{}_adjust_{}_{}.pth'.format(self.source,step,self.net.name))
            best_accuracy = acc

            save_dir = '/home/jake/Gits/AI college/XAI/2.problem/log/' + '{}_adjust_{}_{}.pth'.format(self.source,step,self.net.name)
            print('save_dir->',save_dir)
            if os.path.exists(save_dir):
                os.remove(save_dir)

            save_name = save_dir +'.hdf5'
            print('save_name={}'.format(save_name))
            with h5py.File(save_name, 'w') as hf:
                hf.create_dataset('acc', data=best_accuracy)
                hf.create_dataset('step', data=step)

    def startTrain(self,epoch,trainloader,testloader,step,source):
        print('startTrain')
        for i in range(epoch):
            self.train(i,trainloader,step)
            self.test(i,testloader,step,source)
def starTest(net,testloader,device,step,source):
    print('startTest')
    PATH = '/home/jake/Gits/AI college/XAI/2.problem/model_weights/{}_adjust_{}_{}.pth'.format(source,step,net.name)
    print(PATH)
    net.load_state_dict(torch.load(PATH))
    total = 0
    predicted = 0
    correct =0
    accarr = []
    setparr=[]

    net.eval()



    with torch.no_grad():
        for i, (images, target) in enumerate(testloader):
            inputs , targets = images.to(device), target.to(device)
            outputs = net(inputs)
            total += targets.size(0)
            _,predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
        accarr.append(correct / total)
        setparr.append(step / 10)
        print("test accuracy:{} %% ".format(100. * correct / total))
    return accarr,setparr
