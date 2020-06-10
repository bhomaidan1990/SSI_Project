import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def diceMetric(pred, target, class_num=4, thres=False, per_class=False):
    smooth = 1
    if target.ndim == 3:
        # convert to one hot
        newMask = torch.zeros(target.shape[0], class_num, target.shape[1], target.shape[2])
        for i in range(class_num):
            temp = torch.zeros_like(target)
            temp[target==i] = i
            newMask[:,i,:,:] = temp
        target = newMask.to(target.device)
    pred = F.softmax(pred, dim=1)
    if thres:
        pred[pred>=0.5] = 1
        pred[pred<0.5] = 0

    if per_class:
        dice = torch.zeros(class_num)
        for i in range(class_num):
            tempPred = pred[:,i,:,:]
            tempTarget = target[:,i,:,:]
            pflat = tempPred.contiguous().view(-1)
            mflat = tempTarget.contiguous().view(-1)
            intersection = (pflat * mflat).sum()
            pArea = torch.sum(pflat)
            mArea = torch.sum(mflat)
            dice[i] = 2*intersection/(pArea + mArea)
        
    else:
        pflat = pred.contiguous().view(-1)
        mflat = target.contiguous().view(-1)
        intersection = (pflat * mflat).sum()
        pArea = torch.sum(pflat)
        mArea = torch.sum(mflat)
        dice = 2*intersection/(pArea + mArea)
    return dice

def dice_loss(input, target, class_num=4, per_class=True):
    smooth = 1.
    input = F.softmax(input, 1)

    if per_class:
        device = input.device
        size = input.size()
        dice = torch.zeros(class_num).to(device)
        pflat = input.contiguous().view(size[0], size[1], -1)
        pArea = torch.sum(pflat, dim=2)
        mflat = target.contiguous().view(size[0], size[1], -1)
        mArea = torch.sum(mflat, dim=2)
        intersection = (pflat * mflat).sum(dim=2)
        dice = 2*intersection/(pArea + mArea)
        dice = dice.sum()/(size[0]*size[1])
        loss = 1-dice
    else:
        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        loss = 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

    return loss

class diceLoss(nn.Module):
    
    def __init__(self, coef=0.5):
        super(diceLoss, self).__init__()
        self.coef = coef

    def forward(self,pred,target):
        bce = F.binary_cross_entropy_with_logits(pred, target)
        dice = dice_loss(pred, target)
        return (self.coef*bce + (1-self.coef)*dice)
        # return dice

    
def findArea(img):
    # softmax first
    pred = torch.argmax(img, dim=1, keepdim=True)
    b = img.shape[0]
    c = img.shape[1]
    for i in range(b):
        temp = pred[i,0,:,:]
        for j in range(c):
            tempImg = img[i,j,:,:]
            tempImg[temp!=j] = 0
            tempImg[temp ==j] = 1
            img[i,j,:,:] = tempImg
    # argmax
    area = torch.sum(img, (3,2))
    
    return area.cpu().detach().transpose(0,1).numpy()
    
def TPPerClass(pred, mask, remove_zero=True):
    # shape = pred.shape
    # total = shape[2] * shape[3]
    total = torch.sum(mask, (3,2))
    TP = pred.bool()*mask.bool()
    # FN = (~pred.bool()) * (~mask.bool())
    TP = torch.sum(TP,(3,2))
    
    non_zero_cnt = (total!=0).sum(0)

    total[total==0] = mask.shape[2] * mask.shape[3]
    TP[total==0] = 0
    
    if remove_zero:
        acc = torch.div(TP, total)
        acc = acc.sum(0)
        acc = acc/non_zero_cnt
    else:
        acc = torch.div(TP, total)
        acc = acc.mean(0)    
    # op_shape = TP.shape
    # acc = TP.view(-1)/total.view(-1)
    # acc = acc.view(op_shape)
    
    return acc.cpu().detach().numpy()

def PositivePredictedValue(pred, mask, remove_zero=True):

    total = torch.sum(pred, (3,2))
    TP = pred.bool()*mask.bool()
    
    TP = torch.sum(TP,(3,2))

    non_zero_cnt = (total!=0).sum(0)
    total[total==0] = mask.shape[2] * mask.shape[3]
    TP[total==0] = 0
    
    if remove_zero:
        acc = torch.div(TP, total)
        acc = acc.sum(0)
        acc = acc/non_zero_cnt
    else:
        acc = torch.div(TP, total)
        acc = acc.mean(0)    

    return acc.cpu().detach().numpy()