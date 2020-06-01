import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import keras
import keras.backend as K
from Keras_Data_Loader import DataGenerator
from model import unet, dice_coef, dice_coef_loss
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# import split_folders
# import tqdm
#===========================================================
# Check if there is valid GPU to be used
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print('GPU is Availabele cool...')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    print('Warning ... you are working on CPU, it will take ages!')
#--------------------------------------
# root path to data
dataset = "D:/Saudi_CV/Vibot/Smester_2/5_SSI/SSI_Project/20_Dataset/Dataset"

# split the data
# split_folders.ratio(dataset, output=os.path.join(dataset, 'splited'), seed=1337, ratio=(.8, .1, .1))

trainPath = os.path.join(dataset, 'train/img')
valPath   = os.path.join(dataset, 'val/img')
testPath  = os.path.join(dataset, 'test/img')

trainlabelPath = os.path.join(dataset, 'train/label')
vallabelPath   = os.path.join(dataset, 'val/label')
testlabelPath  = os.path.join(dataset, 'test/label')

#-----------------------------------------------------------------------------------
# Load the Data
#==================
trainGenerator = DataGenerator(trainPath, trainlabelPath, shuffle=True, batch_size=8)
valGenerator   = DataGenerator(valPath, vallabelPath, shuffle=True, batch_size=8)
testGenerator  = DataGenerator(testPath, testlabelPath, shuffle=True, batch_size=8)
#==================

model = unet()
trainEpoch = 5
# model.summary()
#--------------------------------
# Load the saved model if any
#--------------------------------
# model = keras.models.load_model("UNet_Model.h5",compile=False)
# model.compile(optimizer='adam', loss= dice_coef_loss, metrics= [dice_coef])
#---------------------------------------
# Fitting(Training)
#---------------------------------------
history = model.fit_generator(
    trainGenerator,
     steps_per_epoch=len(trainGenerator),
     epochs=trainEpoch,
     validation_data=valGenerator,
     validation_steps=len(valGenerator)//trainEpoch)

#====================
# Save the Rsults
#====================
# Save the model
model.save('UNet_Model.h5')
# # Save the history
np.save('UNet_History.npy',history.history)
history = np.load('UNet_History.npy',allow_pickle='TRUE').item()
#====================
# Plot Fitting curves
#====================

plt.figure(figsize=(30, 5), num = 'Metrics')
plt.subplot(121)
plt.plot(history['dice_coef'])
plt.plot(history['val_dice_coef'])
plt.title('Model accuracy')
plt.ylabel('precision')
plt.xlabel('Epoch')
plt.legend(['Train', 'val'], loc='upper left')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'val'], loc='upper left')

plt.show()

