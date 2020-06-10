import os
from skimage.io import imread
import torch
from torch.utils.data import Dataset
import numpy as np
from matplotlib import pyplot as plt
import skimage
from skimage import transform

def reverse_one_hot(label):
    num_class = label.shape[0]
    result = np.zeros((label.shape[1], label.shape[2]))
    for i in range(num_class):
        result[label[i,:,:]==1] = i
    return result
    
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

def crop2D(img, newSize):
    x = img.shape[-2]
    y = img.shape[-1]
    cropX = x - newSize[0]
    cropY = y - newSize[1]
    if cropX % 2 == 0:
        ex = 0
    else:
        ex = 1
    if cropY % 2 == 0:
        ey = 0
    else:
        ey = 1
    cx = int(cropX/2)
    cy = int(cropY/2)

    return img[:,cx:x-cx-ex,cy:y-cy-ey]

def crop3D(img, newSize):
    x = img.shape[-2]
    y = img.shape[-1]
    cropX = x - newSize[0]
    cropY = y - newSize[1]
    if cropX % 2 == 0:
        ex = 0
    else:
        ex = 1
    if cropY % 2 == 0:
        ey = 0
    else:
        ey = 1
    cx = int(cropX/2)
    cy = int(cropY/2)
    test = img[:, :, cx:x-cx-ex, cy:y-cy-ey]
    return test


class MyDataset(Dataset):
    def __init__(self, img_path, label_path, img_size=48, reverse_one_hot=False, transfrom=None, resize=False):
        self.img_path = img_path
        self.label_path = label_path
        self.img_size = img_size
        self.reverse_one_hot = reverse_one_hot
        self.file_list = []
        self.resize = resize
        self.transfrom = transfrom
        self.loadFile()

    def loadFile(self):
        self.file_list = os.listdir(self.img_path)

    def __getitem__(self, index):
        image_file = os.path.join(self.img_path, self.file_list[index])
        label_file = os.path.join(self.label_path, self.file_list[index])
        image = imread(image_file)

        # image = np.reshape(image, (1, image.shape[0], image.shape[1]))
        if self.img_size[0] == 3:
            image = skimage.color.gray2rgb(image)
            image = np.transpose(image, [2,0,1])
            if self.resize:
                # image = transform.resize(image, [self.img_size[1], self.img_size[2]])
                image = crop2D(image, (self.img_size[1], self.img_size[2]))
            # image = np.transpose(image, [2,0,1])
        else:
            # image = transform.resize(image, self.img_size)
            image = np.expand_dims(image, 0)
            image = crop2D(image, (self.img_size[1], self.img_size[2]))
        image = torch.from_numpy(image).type(torch.FloatTensor)
        label = imread(label_file)
        t = imread(label_file)
        label = np.transpose(label, [2,0,1])
        if self.reverse_one_hot:
            label = reverse_one_hot(label)
        if self.resize:
            if len(label.shape)==2:
                label = np.expand_dims(label, 0)
                label = crop2D(label, (self.img_size[1], self.img_size[2]))
            else:
                label = crop2D(label, (self.img_size[1], self.img_size[2]))
        if self.reverse_one_hot:
            label = torch.from_numpy(label).type(torch.LongTensor)
        else:
            label = torch.from_numpy(label).type(torch.FloatTensor)
        if self.transfrom is not None:
            image = self.transfrom(image)
            # label = self.transfrom(label)
        return image, label

    def __len__(self):
        return len(self.file_list)
