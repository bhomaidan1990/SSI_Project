import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def diceMetric(pred, target, class_num=4, thres=False):
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
    pflat = pred.contiguous().view(-1)
    mflat = target.contiguous().view(-1)
    if thres:
        pflat[pflat>=0.5] = 1
        pflat[pflat<0.5] = 0
    intersection = (pflat * mflat).sum()
    pArea = torch.sum(pflat)
    mArea = torch.sum(mflat)
    dice = 2*intersection/(pArea + mArea)
    return dice

def dice_loss(input, target):
    smooth = 1.
    input = F.softmax(input, 1)
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

class diceLoss(nn.Module):
    
    def __init__(self, coef=1):
        super(diceLoss, self).__init__()
        self.coef = coef

    def forward(self,pred,target):
        bce = F.binary_cross_entropy_with_logits(pred, target)
        dice = dice_loss(pred, target)
        return (self.coef*bce + (1-self.coef)*dice)
        # return dice

    
