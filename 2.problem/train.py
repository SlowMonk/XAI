import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import model
import utils

print("=============train.py=============")

if not os.path.exists('./model'):
    os.mkdir('model/')

train_loader = utils.load_data_cifar10(batch_size=128,test=False)
test_loader = utils.load_data_cifar10(batch_size=100, test=True)
is_cuda = torch.cuda.is_available()
print('cuda available->',is_cuda)

device = torch.device('cuda' if is_cuda else "cpu")
net = model.get_net().to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

min_loss = 999
best_accuracy = 0

def train(epoch):
    for epoch in range(epoch):
        epoch_loss = 0
        min_loss = 999
        total = 0
        correct = 0
        for i , (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs, _ = net(images)
            loss = criterion(outputs,targets)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if (i + 1) % 1000 == 0:
                print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f  correct: %.4f %%'
                      % (epoch + 1, 10, i + 1, len(train_loader), loss.item(),100.*correct/total))

        avg_epoch_loss = epoch_loss / len(train_loader)

        print("Epoch: %d, Avg Loss: %.4f" % (epoch + 1, avg_epoch_loss))


        if avg_epoch_loss < min_loss:
            print("Renew model")
            min_loss = avg_epoch_loss
            print("train saving", min_loss)
            torch.save(net.state_dict(), 'model/cnn.pth')
    print("----------------------------------")
    print("finish trainning")

def test(epoch):
    
train(20)

