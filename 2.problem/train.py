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
    print("training....")
    net.train()
    epoch_loss = 0
    min_loss = 999
    total = 0
    correct = 0
    for i , (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, _ = net(images)
        loss = criterion(outputs,targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (i + 1) % 1000 == 0:
            print('Epoch [%d/%d], Iter [%d/%d], Loss: %.4f  correct: %.4f %%'
                  % (epoch + 1, 10, i + 1, len(train_loader), loss.item(),100.*correct/total))

    avg_epoch_loss = epoch_loss / len(train_loader)

    print("Epoch: %d, Avg Loss: %.4f" % (epoch + 1, avg_epoch_loss))


    #if avg_epoch_loss < min_loss:
    #    print("Renew model")
    #    min_loss = avg_epoch_loss
    #    print("train saving", min_loss)
    #    torch.save(net.state_dict(), 'model/cnn.pth')
    #print("----------------------------------")
    #print("finish trainning")

def test(epoch):
    print("testing....")
    global best_accuracy

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i , (images,target) in enumerate(test_loader):
            inputs , targets = images.to(device) , target.to(device)
            outputs ,_ = net(inputs)
            total += targets.size(0)
            _ ,predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
        print("test accuracy:{} %% ".format(100.*correct/total))
    acc = 100.*correct/total
    if acc > best_accuracy:
        print("saving/////////////////////////////")
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state,'./checkpoint/ckpt.pth')
        torch.save(net.state_dict(), 'model/cnn.pth')
        best_accuracy = acc
start_epoch = 0
for epoch in range(start_epoch, start_epoch+20):
    train(epoch)
    test(epoch)
