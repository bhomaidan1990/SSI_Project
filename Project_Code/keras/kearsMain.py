import keras
import keras.backend as K
from Keras_Data_Loader import DataGenerator
from model import unet
import os
import split_folders
import tqdm

# root path to data
dataset = "/home/dj/Documents/ssProject/20slices/processed/"

# split the data
# split_folders.ratio(dataset, output=os.path.join(dataset, 'splited'), seed=1337, ratio=(.8, .1, .1))

labelPath = os.path.join(dataset, 'label')
trainPath = os.path.join(dataset, 'splited/train/img')
valPath = os.path.join(dataset, 'splited/val/img')
testPath = os.path.join(dataset, 'splited/test/img')

trainEpoch = 5


trainGenerator = DataGenerator(trainPath, labelPath, shuffle=True, batch_size=16)
testGenerator = DataGenerator(testPath, labelPath, shuffle=True, batch_size=16)
valGenerator = DataGenerator(valPath, labelPath, shuffle=True, batch_size=16)


model = unet()

# model.summary()

model.fit_generator(trainGenerator, steps_per_epoch=len(trainGenerator), 
epochs=trainEpoch, validation_data=valGenerator, validation_steps=len(valGenerator))



