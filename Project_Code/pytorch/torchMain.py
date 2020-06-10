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
from torchUtils import *
from matplotlib import pyplot as plt
import skimage
from skimage import transform
import json
import segmentation_models_pytorch as smp
from scipy.stats import pearsonr
from evaluate import bland_altman_plot


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
        dice = diceMetric(pred, target, thres=True, per_class=True)
        dice = dice.detach().cpu().numpy()
        dice_log.append(dice)
        print('\r' + 'batch complete {:.2f}%'\
            .format((batch_idx + 1) / total_len * 100), end='', flush=True)
        train_loss += loss.item()
        loss.backward()
        opt.step()
        # gatehring surface data


    end_time = time.time()
    avg_dice = sum(dice_log)/len(dice_log)
    print('\nElapsed time for training one epoch is %.2f s' % (end_time - start_time))
    print('\r' + 'training loss: {:.2f}'\
            .format(train_loss))
    # if len(avg_dice) == 1:
    #     print('\r' + 'avg dice metric is: {:.3f}'.format(avg_dice))
    # else:
    np.set_printoptions(precision=3)
    print('\r' + 'avg dice metric is: ', avg_dice.mean())
    return float(train_loss), avg_dice.mean()


def test(net, dataset, device):
    net.eval()
    dice_log = []
    dice_thres_log = []
    total_len = len(dataset)
    global_dice_log = []
    PPV_log = None
    precision_log = None
    predArea = None
    maskArea = None
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataset):
            data, target = data.to(device), target.to(device)
            pred = net(data)
            dice = diceMetric(pred, target, thres=True, per_class=True)
            global_dice_log.append(diceMetric(pred, target, thres=True))
            dice = dice.detach().cpu().numpy()
            dice_log.append(dice)
            dice_thres = diceMetric(pred, target, thres=True)
            dice_thres_log.append(dice_thres)
            # record area
            if predArea is None:
                predArea = findArea(pred)
            else:
                predArea = np.hstack((predArea, findArea(pred)))

            if maskArea is None:
                maskArea = findArea(target)
            else:
                maskArea = np.hstack((maskArea, findArea(target)))
            if precision_log is None:
                precision_log = TPPerClass(pred, target)
            else:
                precision_log = (precision_log + TPPerClass(pred, target))/2
            if PPV_log is None:
                PPV_log = PositivePredictedValue(pred, target)
            else:
                PPV_log = (precision_log + PositivePredictedValue(pred, target))/2
            # total_loss += torch.nn.CrossEntropyLoss(output, temp)
            print('\r' + 'test batch complete {:.2f}% '.format((batch_idx + 1) / total_len * 100), end='', flush=True)
    avg1 = sum(dice_log)/len(dice_log)
    avg2 = sum(dice_thres_log)/len(dice_thres_log)
    # print('\r' + 'test avg dice metric(no threshold): {:.4f}\ntest avg dice metric(threshold): {:.4f}'\
            # .format(avg1, avg2))
    np.set_printoptions(precision=3)
    avg_dice = sum(dice_log)/len(dice_log)
    
    # pearson correlation of area for different class
    print('\n')
    for i in range(4):
        pA = predArea[i,:]
        pM = maskArea[i,:]
        [c, p] = pearsonr(pA, pM)
        print('correlation for class {:d} area is: {:.4f}, p-value is: {:.4f}'.format(i, c, p))

    # print('\nAvg precision (TP+FN) per class is: ', precision_log)

    print('\nAvg TP rate per class is: ', precision_log)

    print('\nAvg Positive Predicted Value is: ', PPV_log)

    print('\nGlobal dice metrix is: {:.4f}'.format(sum(global_dice_log)/len(global_dice_log)))

    # show bland altman plot for all class
    # fig, axes = plt.subplots(2,2)
    # for i in range(4):
    #     idx = int(i/2)
    #     tempAx = axes[idx,i%2]
    #     bland_altman_plot(predArea[i,:]/(224*224), maskArea[i,:]/(224*224), ax=tempAx, area=True)
        
    #     tempAx.set_title('area for class ' + str(i))
    # plt.show()

    print('\nweighted avg dice = ', avg_dice)
    print('Avg for all class = ', avg_dice.sum()/4, '\n')
    return avg1, avg2


if __name__ == "__main__":
    # hyper parameters
    acc = 0
    lr = 1e-3
    trainEpoch = 1
    batch_size = 16
    num_worker = 6
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
    # mobilenet from segmentation model library
    # ======================================================
    # net = mobileUnet(torchvision.models.mobilenet_v2())
    # modelName = 'mobileNet_freeze_se_BCE_and_Dice_per_class_loss_Adam_'
    # net.freeze_encoder()
    # net.initialize_decoder()


    # ======================================================
    # efficientNet b7 from segmentation model library
    # ======================================================
    net = smp.Unet('efficientnet-b7', classes = 4)
    modelName = 'efficientnet-b7_half_dice_per_class_'

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

    loss = diceLoss(coef=0)


    # train_set = MyDataset(train_path, labelPath, img_size=[3,224,224], transfrom=None, resize=True)
    # test_set = MyDataset(test_path, labelPath, img_size=[1,256,256], resize=True, reverse_one_hot=True)

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
        pretrained = torch.load(save_dir + modelName + 'model.tar')
        net.load_state_dict(pretrained['model'])
        # net.load_state_dict(torch.load(save_dir + modelName + 'model.tar'))
        print('loading the pre-trained weights')
    else:
        pretrained = {'acc':0, 'model':net.state_dict()}

    # ======================================================
    # choose optimizer, use filter when freeze the encoder
    # ======================================================
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.8)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    # optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.97)

    loss_log = []
    train_dice_log = []
    test_dice_log = []
    test_dice_thres_log = []
    test_dice_per_class = []
    train_dice_per_class = []
    # save_dict = {'acc':0, 'model':net.state_dict()}
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for i in range(trainEpoch):
        print('\n' + '=='*10)
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        print('Training at epoch {:d} out of {:d}, current lr = {:.6f}'.format(i+1, trainEpoch, current_lr))

        print('Current model: ', modelName + 'model')
        
        # l, d = train(net, train_loader, optimizer, device, loss)
        # scheduler.step()
        # loss_log.append(l)
        # train_dice_per_class.append(d)
        # train_dice_log.append(d.mean())

        avg1, avg2 = test(net, test_loader, device)
        
        # test_dice_log.append(avg1)
        test_dice_thres_log.append(avg2)
        test_acc = avg1.sum()/4
        test_dice_log.append(test_acc)
        test_dice_per_class.append(avg1)
        if test_acc > pretrained['acc']:
            pretrained['acc'] = test_acc
            pretrained['per_class'] = avg1
            pretrained['model'] = net.state_dict()
            torch.save(pretrained, save_dir + modelName + 'model.tar')
            print('saving model params...')
        print('=='*10)
        print('best acc so far: ', pretrained['acc'], 'per_class: ', pretrained['per_class'])
    
    # log_dict = {}
    # log_dict['train_loss'] = loss_log
    # log_dict['train_dice'] = train_dice_log
    # log_dict['test_dice'] = test_dice_log
    # # json.dump(log_dict, open(modelName + 'log.json', 'w'))
    # log_path =os.path.join(strProjectFolder, modelName + 'log')
    # if os.path.isdir(log_path):
    #     pass
    # else:
    #     os.mkdir(log_path)
    
    # print('saving training log')
    # np.save(os.path.join(log_path, 'train_loss') ,np.array(loss_log))
    # np.save(os.path.join(log_path, 'train_dice') ,np.array(train_dice_log))
    # np.save(os.path.join(log_path, 'test_dice') ,np.array(test_dice_log))
    # np.save(os.path.join(log_path, 'test_dice_per_class') ,np.array(test_dice_per_class))
    # np.save(os.path.join(log_path, 'train_dice_per_class') ,np.array(train_dice_per_class))

    # print('finish training, processing visualization...')
    # plt.subplot(121)
    # plt.plot(range(len(loss_log)), loss_log, 'r-')
    # plt.title('training loss')
    # plt.subplot(122)
    # plt.plot(range(len(train_dice_log)), train_dice_log, 'r-', label='train')
    # plt.plot(range(len(test_dice_log)), test_dice_log, 'b-', label='test')
    # plt.legend(loc='lower right')
    # plt.title('dice')
    # plt.show()
