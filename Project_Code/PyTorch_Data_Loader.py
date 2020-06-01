#================================
#===  @Author: Belal Hmedan  ====
#================================
##      PyTorch DataLoader     ##
#--------------------------------
#============================
# Import necessary libs
#============================
import os
import numpy as np
import glob
from skimage.io import imread
#------------------------------
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
#------------------------------
# class to load the dataset
class PNGDataset(Dataset):
    #'Loads the data for PyTorch'
    def __init__(self, images_path, labels_path,transform=None,image_dimensions = (256,256),
        n_channels=1,n_classes = 4, shuffle=False, augment=False):

        self.images_path  = images_path                  # images dir
        self.labels_path  = labels_path                  # labels dir
        self.images_path_list  = glob.glob(images_path+'*.png')    # images paths
        self.labels_path_list  = glob.glob(labels_path+'*.png')    # labels paths

        self.transform    = transform                    # Transformations 
        self.dim          = image_dimensions             # image dimensions
        self.n_channels   = n_channels                   # number of channels
        self.n_classes    = n_classes                    # number of classes
        self.shuffle      = shuffle                      # shuffle bool
        self.augment      = augment                      # augment data bool
        self.on_epoch_end()

    def __len__(self):
        #'Denotes the number of the images
        return len(self.images_path_list)

    def on_epoch_end(self):
        #'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.images_path_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # select data and load images
        images = np.expand_dims(imread(self.images_path_list[index]),axis=2)
        labels = imread(self.labels_path_list[index])

        # preprocess and augment data
        if self.augment == True:
            images = self.augmentor(images)
            labels = self.augmentor(labels)
        # in case BACKBONE preprocessing needed 
        # global preprocess_input
        # images = np.array([preprocess_input(img) for img in images])
        # labels = np.array([preprocess_input(img) for img in labels])
        sample = images, labels
        # Apply Transformation 
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    
    def augmentor(self, images):
        #'Apply data augmentation'
        # Our Augmentation can be here or seperate!
        return images
#----------------------------------------------------
# class to transform images from arrays to Tensors
class ToTensor:
    def __call__(self,sample):
        images, labels = sample
        return torch.from_numpy(images), torch.from_numpy(labels)

#----------------------------------------------------
# Example
#===========
# Don't forget to change your path !!!
#-------------------------------------
# images_path = 'D:/Saudi_CV/Vibot/Smester_2/5_SSI/SSI_Project/20_Dataset/PNGDataset/img/'
# labels_path = 'D:/Saudi_CV/Vibot/Smester_2/5_SSI/SSI_Project/20_Dataset/PNGDataset/label/'

# composed = torchvision.transforms.Compose([ToTensor()])
# dataset = PNGDataset(images_path, labels_path, transform=composed)
# dataloader = DataLoader(dataset=dataset,batch_size=8, shuffle=True, num_workers=0,drop_last=True )

# images, labels = next(iter(dataloader))
# print(images.shape,labels.shape)

# if __name__ == '__main__':
#     pass
#     # for a in dataloader:
#     #     print(a[0].shape,a[1].shape)

#     data = next(iter(dataloader))
#     images, labels = data
#     print(images.shape, labels.shape)