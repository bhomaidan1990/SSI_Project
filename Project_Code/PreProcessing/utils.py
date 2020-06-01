#================================
#===  @Author: Deng Jianning ====
#================================
"""
This file contains utility functions that will be used 
"""

import nibabel as nib
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imsave

def imgNorm(img):
    div = np.max(img) - np.min(img)
    
    return 255*(img - np.min(img))/div


def oneHotMask(mask, cataNum=4):
    shape = mask.shape
    shape = list(shape)
    shape.append(cataNum)
    result = np.zeros(shape)
    # print(np.max(mask))
    for i in range(cataNum):
        temp = np.zeros(mask.shape)
        temp[mask==i] = 1
        result[:,:,i] = temp
    return result

def pre_processing(imgPath, gtPath, opPath, normalize=True, oneHot=True, newSize=[256,256], cataNum=4):
    """
    the ouput png file will be store in folder 'img' and 'label' under given opPath

    @param imgPath: folder containing the input nifti file
    @param gtPath: folder contatining the groud truth nifti file
    @param opPath: output folder
    @param normalize: normalize the input or not
    @param oneHot: convert the mask to one hot image or not
    @param newSize: the size of the ouput img/label
    @param cataNum: number of catagories for mask
    
    example:
    imgPath = './COVID-19-CT-Seg_20cases/'
    gtPath = './Lung_and_Infection_Mask/'
    savePath = './processed/'
    pre_processing(imgPath, gtPath, savePath)

    """
    imgFile = imgPath
    gtFile = gtPath
    savePath = opPath

    gtList = os.listdir(gtFile)

    opImgPath = os.path.join(savePath,'img/')
    opGtPath = os.path.join(savePath,'label/')


    if not os.path.isdir(savePath):
        os.mkdir(savePath)
    if not os.path.isdir(opImgPath):
        os.mkdir(opImgPath)
    if not os.path.isdir(opGtPath):
        os.mkdir(opGtPath)

    studyNum = len(gtList)

    for i in range(studyNum):
        gt = nib.load(os.path.join(gtFile , gtList[i])).get_fdata()
        img = nib.load(os.path.join(imgFile, gtList[i])).get_fdata()
        
        slices = img.shape[2]
        for s in range(slices):
            if oneHot:
                gtTemp = oneHotMask(gt[:,:,s], cataNum=cataNum)
            else:
                gtTemp = gt[:,:,s]
            gtTemp = resize(gtTemp, newSize, preserve_range=True)
            gtTemp = gtTemp.astype(np.uint8)
            imgTemp = resize(img[:,:,s], newSize, preserve_range=True)
            imgTemp = imgNorm(imgTemp)
            imgTemp = imgTemp.astype(np.uint8)
            gtName = 'case_' + str(i+1) + '_slices_' + str(s+1) + '.png'
            imgName = 'case_' + str(i+1) + '_slices_' + str(s+1) + '.png'
            imsave(opImgPath+imgName, imgTemp, check_contrast=False)
            imsave(opGtPath+gtName, gtTemp, check_contrast=False)

    return