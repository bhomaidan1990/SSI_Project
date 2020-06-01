import torch
import torch.optim as optim
import torchvision
from torchModel import UNet, mobileUnet
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from torchDataloader import MyDataset
from torch.autograd import Variable
import graphviz
import torchviz
from torchUtils import diceMetric, diceLoss, dice_loss
from matplotlib import pyplot as plt
import skimage
from skimage import transform

import segmentation_models_pytorch as smp


def train(net, data_loader, opt, device, criterion):
    net.train()
    train_loss = 0
    total_len = len(data_loader)
    start_time = time.time()
    dice_log = []
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        pred = net(data)
        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            loss = criterion(pred, target.long())
        else:
            loss = criterion(pred, target)
        dice = diceMetric(pred, target, thres=True)
        dice_log.append(float(dice))
        print('\r' + 'batch complete {:.2f}%'\
            .format((batch_idx + 1) / total_len * 100), end='', flush=True)
        train_loss += loss.item()
        loss.backward()
        opt.step()

    end_time = time.time()
    print('\nElapsed time for training one epoch is %.2f' % (end_time - start_time))
    print('\r' + 'training loss: {:.2f}, avg dice metric: {:.4f}'\
            .format(train_loss, sum(dice_log)/len(dice_log)))
    return float(train_loss), sum(dice_log)/len(dice_log)


def test(net, dataset, device):
    net.eval()
    dice_log = []
    dice_thres_log = []
    total_len = len(dataset)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataset):
            data, target = data.to(device), target.to(device)
            pred = net(data)
            dice = diceMetric(pred, target)
            dice_log.append(dice)
            dice_thres = diceMetric(pred, target, thres=True)
            dice_thres_log.append(dice_thres)
            # total_loss += torch.nn.CrossEntropyLoss(output, temp)
            print('\r' + 'test batch complete {:.2f}% '.format((batch_idx + 1) / total_len * 100), end='', flush=True)
    avg1 = sum(dice_log)/len(dice_log)
    avg2 = sum(dice_thres_log)/len(dice_thres_log)
    print('\r' + 'test avg dice metric(no threshold): {:.4f}\nteset avg dice metric(threshold): {:.4f}'\
            .format(avg1, avg2))

    return avg1, avg2


if __name__ == "__main__":
    # hyper parameters
    acc = 0
    lr = 1e-4
    trainEpoch = 20
    batch_size = 16
    num_worker = 8
    # prepare data
    strProjectFolder = os.path.dirname(__file__)
    # strProjectFolder = "/home/dj/Documents/ssProject/20slices/pytorch/"
    print('file folder = ',strProjectFolder)
    dataset = "/home/dj/Documents/ssProject/20slices/processed/"
    
    labelPath = os.path.join(dataset, 'label')
    train_path = os.path.join(dataset, 'splited/train/img')
    test_path = os.path.join(dataset, 'splited/test/img')

    # prepare model
    save_dir = strProjectFolder + '/save/'
    

    # ======================================================
    # custom model with mobileNet
    # ======================================================
    # net = mobileUnet(torchvision.models.mobilenet_v2())
    # modelName = 'mobileNet'
    # net.freeze_encoder()
    # net.initialize_decoder()


    # ======================================================
    # efficientNet b7 from segmentation model library
    # ======================================================
    net = smp.Unet('efficientnet-b7', classes = 4)
    modelName = 'efficientnet-b7'

    # freeze encoder
    for param in net.encoder.parameters():
        param.requires_grad=False


    # ======================================================
    # basic Unet
    # ======================================================
    # net = UNet(num_classes=4, input_channel=3)
    # modelName = 'Unet'



    # ======================================================
    # select different preprocessing configuration
    # reverse one hot label when using cross entropy loss
    # ======================================================

    # train_set = MyDataset(train_path, labelPath, img_size=[3,256,256], transfrom=None, resize=True, 
    # reverse_one_hot=False)
    # loss = diceLoss()

    # train_set = MyDataset(train_path, labelPath, img_size=[3,256,256], transfrom=None, resize=True, 
    # reverse_one_hot=True)
    # loss = torch.nn.CrossEntropyLoss()

    train_set = MyDataset(train_path, labelPath, img_size=[3,224,224], transfrom=None, resize=True, 
    reverse_one_hot=False)
    test_set = MyDataset(test_path, labelPath, img_size=[3,224,224], transfrom=None, resize=True, 
    reverse_one_hot=False)
    loss = diceLoss()


    # train_set = MyDataset(train_path, labelPath, img_size=[3,224,224], transfrom=None, resize=True)
    # test_set = MyDataset(test_path, labelPath, img_size=[1,256,256], resize=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_worker, 
    pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_worker, 
    pin_memory=True)
    
    

    # ======================================================
    # model visualization
    # ======================================================
    # dummpy_input = np.zeros([1,1,256,256])
    # dummpy_input = Variable(torch.rand(1,1,256,256))
    # y = net(dummpy_input)
    # graph = torchviz.make_dot(y.mean(), params=dict(net.named_parameters()))
    # graph.render('graph', view=True, format='png')
    # graph.view()
    

    # load pre-trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        torch.cuda.empty_cache() 
    net.to(device)
    if os.path.isfile(save_dir + modelName +'model.tar'):
        net.load_state_dict(torch.load(save_dir + modelName + 'model.tar'))
        print('loading the pre-trained weights')

    # ======================================================
    # choose optimizer, use filter when freeze the encoder
    # ======================================================
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.8)
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

    loss_log = []
    train_dice_log = []
    test_dice_log = []
    test_dice_thres_log = []
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for i in range(trainEpoch):
        l, d = train(net, train_loader, optimizer, device, loss)
        loss_log.append(l)
        train_dice_log.append(d)
        torch.save(net.state_dict(), save_dir + modelName + 'model.tar')
        avg1, avg2 = test(net, test_loader, device)
        test_dice_log.append(avg1)
        test_dice_thres_log.append(avg2)
    
    print('finish training, processing visualization...')
    plt.subplot(121)
    plt.plot(range(len(loss_log)), loss_log, 'r-')
    plt.title('training loss')
    plt.subplot(122)
    plt.plot(range(len(train_dice_log)), train_dice_log, 'r-')
    plt.plot(range(len(test_dice_log)), test_dice_log, 'b-')
    plt.title('dice')
    plt.show()
