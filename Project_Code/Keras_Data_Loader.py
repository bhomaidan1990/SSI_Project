#================================
#===  @Author: Belal Hmedan  ====
#================================
##   Flexible data generator   ##
#--------------------------------
# Reference:
# https://www.kaggle.com/mpalermo/keras-pipeline-custom-generator-imgaug
#--------------------------------------------------------------------------
#============================
# Import necessary libs
#============================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import keras
from skimage.io import imread
#----------------------------------------------------------
# # in case of different BackBones # # 
# from keras.applications.resnet50 import preprocess_input
#----------------------------------------------------------
class DataGenerator(keras.utils.Sequence):
    #'Generates data for Keras'
    def __init__(self, images_path, labels_path, batch_size=64,
     image_dimensions = (256 ,256 ),n_channels=1,n_classes = 4, shuffle=False, augment=False):

        self.images_path  = images_path
        self.labels_path  = labels_path
        self.images_list = os.listdir(images_path)       # images path
        self.labels_list = os.listdir(labels_path)       # labels path

        self.dim          = image_dimensions             # image dimensions
        self.n_channels   = n_channels                   # number of channels
        self.n_classes    = n_classes                    # number of classes
        self.batch_size   = batch_size                   # batch size
        self.shuffle      = shuffle                      # shuffle bool
        self.augment      = augment                      # augment data bool
        self.on_epoch_end()

    def __len__(self):
        #'Denotes the number of batches per epoch'
        return int(np.floor(len(self.images_list) / self.batch_size))

    def on_epoch_end(self):
        #'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.images_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        #'Generate one batch of data'
        # selects indices of data for next batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # select data and load images
        images = np.stack([np.expand_dims(imread(os.path.join(self.labels_path,self.labels_list[k])),axis=2) for k in indexes],axis=0)
        labels = np.stack([imread(os.path.join(self.images_path,self.images_list[k])) for k in indexes],axis=0)
        
        
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
datagenerator = DataGenerator(images_path, labels_path)
images, labels = next(iter(datagenerator))
print(images.shape,labels.shape)