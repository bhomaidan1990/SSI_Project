#================================
#===  @Author: Belal Hmedan  ====
#================================
##   Flexible data generator   ##
#--------------------------------
# Reference:
# https://www.kaggle.com/mpalermo/keras-pipeline-custom-generator-imgaug
#--------------------------------------------------------------------------
# Import necessary Libraries
import numpy as np
import keras
from skimage.io import imread
#----------------------------------------------------------
# # in case of different BackBones # # 
# from keras.applications.resnet50 import preprocess_input
#----------------------------------------------------------
class DataGenerator(keras.utils.Sequence):
    #'Generates data for Keras'
    def __init__(self, images_paths, labels_paths, batch_size=64, image_dimensions = (96 ,96 ,3), shuffle=False, augment=False):
        self.labels_paths = labels_paths        # array of label paths
        self.images_paths = images_paths        # array of image paths
        self.dim          = image_dimensions    # image dimensions
        self.batch_size   = batch_size          # batch size
        self.shuffle      = shuffle             # shuffle bool
        self.augment      = augment             # augment data bool
        self.on_epoch_end()

    def __len__(self):
        #'Denotes the number of batches per epoch'
        return int(np.floor(len(self.images_paths) / self.batch_size))

    def on_epoch_end(self):
        #'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.images_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        #'Generate one batch of data'
        # selects indices of data for next batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        # select data and load images
        labels = [imread(self.labels_paths[k]) for k in indexes]
        images = [imread(self.images_paths[k]) for k in indexes]
        
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