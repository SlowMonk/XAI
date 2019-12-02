import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import model
import utils
import numpy as np

print("=============train.py=============")

if not os.path.exists('./model'):
    os.mkdir('model/')

train_loader = utils.load_data_cifar10(batch_size=128,test=False)
test_loader = utils.load_data_cifar10(batch_size=100, test=True)
is_cuda = torch.cuda.is_available()
print('cuda available->',is_cuda)

device = torch.device('cuda' if is_cuda else "cpu")
net = model.get_net().to(device)

def test():
    net.eval()
    total_correct = 0
    total_images = 0
    confusion_matrix = np.zeros([10, 10], int)
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs,_ = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            for i, l in enumerate(labels):
                confusion_matrix[l.item(), predicted[i].item()] += 1

    model_accuracy = total_correct / total_images * 100
    print('Model accuracy on {0} test images: {1:.2f}%'.format(total_images, model_accuracy))
#train(20)
test()
