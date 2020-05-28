#================================
#===  @Author: Belal Hmedan  ====
#================================
##      PyTorch DataLoader     ##
#--------------------------------
#============================
# Import necessary libs
#============================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
from skimage.io import imread
#------------------------------
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
#------------------------------

class PNGDataset(Dataset):
    #'Loads the data for PyTorch'
    def __init__(self, images_path, labels_path,image_dimensions = (256 ,256 ),
        n_channels=1,n_classes = 4, shuffle=False, augment=False):

        self.images_path  = images_path
        self.labels_path  = labels_path
        self.images_list = os.listdir(images_path)       # images path
        self.labels_list = os.listdir(labels_path)       # labels path

        self.dim          = image_dimensions             # image dimensions
        self.n_channels   = n_channels                   # number of channels
        self.n_classes    = n_classes                    # number of classes
        self.shuffle      = shuffle                      # shuffle bool
        self.augment      = augment                      # augment data bool
        self.on_epoch_end()

    def __len__(self):
        #'Denotes the number of batches per epoch'
        return len(self.images_list)

    def on_epoch_end(self):
        #'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.images_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # select data and load images
        images = torch.from_numpy(np.expand_dims(imread(os.path.join(self.images_path,self.images_list[index])),axis=2))
        labels = torch.from_numpy(imread(os.path.join(self.labels_path,self.labels_list[index])))
         
        # preprocess and augment data
        if self.augment == True:
            images = self.augmentor(images)
            labels = self.augmentor(labels)
        # in case BACKBONE preprocessing needed 
        # images = np.array([preprocess_input(img) for img in images])
        # labels = np.array([preprocess_input(img) for img in labels])
        return images, labels
    
    
    def augmentor(self, images):
        #'Apply data augmentation'
        # Our Augmentation can be here or seperate!
        return images
#----------------------------------------------------
# Example
#===========
# Don't forget to change your path !!!
#-------------------------------------
images_path = 'D:/Saudi_CV/Vibot/Smester_2/5_SSI/SSI_Project/20_Dataset/Deng_Dataset/img'
labels_path = 'D:/Saudi_CV/Vibot/Smester_2/5_SSI/SSI_Project/20_Dataset/Deng_Dataset/label'

dataset = PNGDataset(images_path, labels_path)
dataloader = DataLoader(dataset=dataset,batch_size=8, shuffle=True, num_workers=1)

if __name__ == '__main__':
    pass
    # for a in dataloader:
    #     print(a[0].shape,a[1].shape)

    data = next(iter(dataloader))
    images, labels = data
    print(images.shape, labels.shape)