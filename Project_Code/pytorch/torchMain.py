import torch
import torch.optim as optim
from torchModel import UNet
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from torchDataloader import MyDataset

def train(net, data_loader, opt, device, criterion):
    net.train()
    correct = 0
    total = 0
    train_loss = 0
    total_len = len(data_loader)
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        # _, predict = output.max(1)
        # total += target.size(0)
        # _, temp = target.max(1)
        # correct += predict.eq(temp).sum().item()
        print('\r' + 'batch complete {:.2f}% '.format((batch_idx + 1) / total_len * 100), end='', flush=True)
        loss = criterion(output, target)
        train_loss += loss.item()
        loss.backward()
        opt.step()

    end_time = time.time()
    print('Elapsed time for training one epoch is %.2f' % (end_time - start_time))
    print('training loss is: %.4f \n' % train_loss)
    return


def test(net, dataset, acc, device, save_dir):
    net.eval()
    correct = 0
    total = 0
    total_loss = 0
    total_len = len(dataset)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataset):
            data, target = data.to(device), target.to(device)
            output = net(data)
            total += target.size(0)
            _, predict = output.max(1)
            _, temp = target.max(1)
            correct += predict.eq(temp).sum().item()
            # total_loss += torch.nn.CrossEntropyLoss(output, temp)
            print('\r' + 'batch complete {:.2f}% '.format((batch_idx + 1) / total_len * 100), end='', flush=True)

    print('accuracy at this epoch: %.3f%%'% (correct * 100 / total))
    if correct/total > acc:
        # print('best accuray: %.3f \n'% (correct / total))
        torch.save(net.state_dict(), save_dir + 'model.tar')
        return correct/total
    else:
        # print('best accuray: %.3f \n'% acc)
        return acc


if __name__ == "__main__":
    # hyper parameters
    acc = 0
    lr = 1e-3
    epoch = 50
    batch_size = 32
    num_worker = 0
    # prepare data
    strProjectFolder = os.path.dirname(__file__)
    dataset = "/home/dj/Documents/ssProject/20slices/processed/"
    
    labelPath = os.path.join(dataset, 'label')
    train_path = os.path.join(dataset, 'splited/train/img')
    test_path = os.path.join(dataset, 'splited/test/img')

    train_set = MyDataset(train_path, labelPath, img_size=[1,256,256])
    test_set = MyDataset(test_path, labelPath, img_size=[1,256,256])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_worker)

    # prepare model
    save_dir = strProjectFolder + '/save/'
    net = UNet(num_classes=4, input_channel=1)
    
    # load pre-trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        torch.cuda.empty_cache() 
    net.to(device)
    if os.path.isfile(save_dir + 'model.tar'):
        net.load_state_dict(torch.load(save_dir + 'model.tar'))
    loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.8)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for i in range(epoch):
        train(net, train_loader, optimizer, device, loss)
        # acc = test(net, test_loader, acc, device, save_dir)
        # print('\nbest accuracy at epoch %d is : %.3f%% \n' % (i, acc*100))
    print('finish training, processing visualization...')


