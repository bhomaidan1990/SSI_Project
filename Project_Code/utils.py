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


def pre_processing(imgPath, gtPath, opPath, normalize=True):
    """
    the ouput png file will be store in folder 'img' and 'label' under given opPath

    @param imgPath: folder containing the input nifti file
    @param gtPath: folder contatining the groud truth nifti file
    @param opPath: output folder
    @param normalize: normalize the input or not

    example:
    from utils import pre_processing
    imgPath = './COVID-19-CT-Seg_20cases/'
    gtPath = './Lung_and_Infection_Mask/'
    savePath = './processed/'
    pre_processing(imgPath, gtPath, savePath)

    """
    imgFile = imgPath
    gtFile = gtPath
    savePath = opPath

    gtList = os.listdir(gtFile)

    opImgPath = os.path.jpoin(savePath,'img/')
    opGtPath = os.path.join(savePath,'label/')


    if not os.path.isdir(savePath):
        os.mkdir(savePath)
    if not os.path.isdir(opImgPath):
        os.mkdir(opImgPath)
    if not os.path.isdir(opGtPath):
        os.mkdir(opGtPath)

    studyNum = len(gtList)
    newSize = [256,256]

    for i in range(studyNum):
        gt = nib.load(gtFile + gtList[i]).get_fdata()
        img = nib.load(imgFile + gtList[i]).get_fdata()
        
        slices = img.shape[2]
        for s in range(slices):
            gtTemp = resize(gt[:,:,s], newSize, preserve_range=True)
            # gtTemp = imgNorm(gtTemp)
            gtTemp = gtTemp.astype(np.uint8)
            imgTemp = resize(img[:,:,s], newSize, preserve_range=True)
            imgTemp = imgNorm(imgTemp)
            imgTemp = imgTemp.astype(np.uint8)
            gtName = 'case_' + str(i) + '_slices_' + str(s) + '_gt.png'
            imgName = 'case_' + str(i) + '_slices_' + str(s) + '.png'
            imsave(opImgPath+imgName, imgTemp, check_contrast=False)
            imsave(opGtPath+gtName, gtTemp, check_contrast=False)

    return

