"""
This file is for model implementation.
It has Convolution layers and Average pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision import models



def weight_init__MnistCNN(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_MnistCNN():
    net = CNN_mnist()
    net.apply(weight_init__MnistCNN)
    print("INIT NETWORK")
    return net


def load__MnistCNN():
    net = CNN_mnist()
    net.load_state_dict(torch.load("/home/jake/Gits/AI college/XAI/2.problem/model_weights/cnn_mnist.pth"))
    return net

def load__Resnet50():
    net = ResNet50()
    net.load_state_dict(torch.load("/home/jake/Gits/AI college/XAI/2.problem/model_weights/resnet50.pth"))
    return net

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_net():
    net = CNN()
    net.apply(weight_init)
    print("INIT NETWORK")
    return net


def load_net():
    net = CNN()
    net.load_state_dict(torch.load("/home/jake/Gits/AI college/XAI/2.problem/model_weights/cnn.pth"))
    return net

def get_resnet50():
    net = ResNet50()
    net.apply(weight_init)
    print('resnet50 network')
    return net
#===========================================Basic cnn=======================================================

class CNN(nn.Module):
    """
    Simple CNN NETWORK
    """

    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            # 3 x 128 x 128
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            # 32 x 128 x 128
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # 64 x 128 x 128
            nn.MaxPool2d(2, 2),

            # 64 x 64 x 64
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # 128 x 64 x 64
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # 256 x 64 x 64
            nn.MaxPool2d(2, 2),

            # 256 x 32 x 32
            nn.Conv2d(256, 10, 3, 1, 1),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.2)
        )
        # 256 x 32 x 32
        self.avg_pool = nn.AvgPool2d(7)
        # 256 x 1 x 1
        self.classifier = nn.Linear(10, 10)

    def forward(self, x):
        features = self.conv(x)
        flatten = self.avg_pool(features).view(features.size(0), -1)
        output = self.classifier(flatten)
        return output, features


#===========================================cnn_mnist=======================================================

class CNN_mnist(nn.Module):
    """
    Simple CNN NETWORK
    """

    def __init__(self):
        super(CNN_mnist, self).__init__()
        self.conv = nn.Sequential(
            # 3 x 128 x 128
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            # 32 x 128 x 128
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            # 64 x 128 x 128
            nn.MaxPool2d(2, 2),

            # 64 x 64 x 64
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            # 128 x 64 x 64
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            # 256 x 64 x 64
            nn.MaxPool2d(2, 2),

            # 256 x 32 x 32
            nn.Conv2d(256, 10, 3, 1, 1),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(0.2)
        )

        # 256 x 32 x 32
        self.avg_pool = nn.AvgPool2d(7)
        # 256 x 1 x 1
        self.classifier = nn.Linear(10, 10)
        self.name='CNN_mnist'

    def forward(self, x):
        features = self.conv(x)
        flatten = self.avg_pool(features).view(features.size(0), -1)
        output = self.classifier(flatten)
        return output, features

#===========================================resnet=======================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


#original_model = models.ResNet(pretrained=True)

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])