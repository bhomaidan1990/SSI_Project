import os
from skimage.io import imread
import torch
from torch.utils.data import Dataset
import numpy as np

def reverse_one_hot(label):
    num_class = label.shape[2]
    result = np.zeros((label.shape[0], label.shape[1]))
    for i in range(num_class):
        result[label[:,:,i]==1] = i
    return result



class MyDataset(Dataset):
    def __init__(self, img_path, label_path, img_size=48):
        self.img_path = img_path
        self.label_path = label_path
        self.img_size = img_size
        self.file_list = []
        self.loadFile()

    def loadFile(self):
        self.file_list = os.listdir(self.img_path)

    def __getitem__(self, index):
        image_file = os.path.join(self.img_path, self.file_list[index])
        label_file = os.path.join(self.label_path, self.file_list[index])
        image = torch.from_numpy(imread(image_file)).view(self.img_size).type(torch.FloatTensor)
        label = imread(label_file)
        label = reverse_one_hot(label)
        label = torch.from_numpy(label).type(torch.LongTensor)
        return image, label

    def __len__(self):
        return len(self.file_list)
