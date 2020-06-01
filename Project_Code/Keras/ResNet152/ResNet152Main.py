#--------------------------------
import os, time, shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import keras 
import tensorflow as tf
from skimage.transform import resize
from skimage.io import imsave
#--------------------------------
import nibabel as nib
import segmentation_models as sm
# import split_folders as sf
#--------------------------------
from Keras_Data_Loader import DataGenerator
#--------------------------------------------------------------
#--------------------------------------
"""### Initialization"""
#--------------------------------------
# For reproducabilty
seed = 42
np.random.seed = seed
tf.random.set_seed = seed
# define BackBone
BACKBONE = 'resnet152'
#--------------------------------------
# Check if there is valid GPU to be used
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print('GPU is Availabele cool...')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    print('Warning ... you are working on CPU, it will take ages!')
#------------------------------------------------------------------
# load your data
dataset = "Dataset"
# Path
train_images_path = os.path.join(dataset,'train/img/')
train_labels_path = os.path.join(dataset,'train/label/')

val_images_path   = os.path.join(dataset,'val/img/')
val_labels_path   = os.path.join(dataset,'val/label/')

test_images_path  = os.path.join(dataset,'test/img/')
test_labels_path  = os.path.join(dataset,'test/label/')
#--------------------------------------------------------------
train_generator      = DataGenerator(train_images_path, train_labels_path)
val_generator        = DataGenerator(val_images_path, val_labels_path)
test_generator_eval  = DataGenerator(test_images_path, test_labels_path)
test_generator       = DataGenerator(test_images_path, test_labels_path,mode='test')
#---------------------------------------------------------------
"""## Define The Model"""

"""Define the model"""
from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.metrics import iou_score
from keras.layers import Input, Conv2D
from keras.models import Model

# define number of channels
num_channels = 1

base_model = Unet(BACKBONE, classes=4, activation='softmax',
    encoder_weights='imagenet', encoder_freeze=False)
# ---------------------------------------------------------------
# In Case The Input is Grayscale
#----------------------------------------------------------------
inp = Input(shape=(None, None, num_channels))
l1 = Conv2D(3, (1, 1))(inp) # map N channels data to 3 channels
out = base_model(l1)
model = Model(inp, out, name=base_model.name)
#----------------------------------------------------------------
# Loss #
#=======
# loss = sm.losses.CategoricalCELoss()
loss = sm.losses.cce_dice_loss
# loss = sm.losses.CategoricalFocalLoss()
#----------------------------------------------------------------
opt = keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999)
# model = keras.models.load_model("drive/My Drive/Dataset/Covid_19/ResNet152_Model.h5",compile=False)
# Compile Model
model.compile(
    opt,
    loss=loss,
    metrics=[sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)],
)
# model.summary()

"""## Fitting(Training)"""

# Model Fitting (Training)
#=========================
batch_size = 8
# Model Checkpoint
Callbacks = [            
             keras.callbacks.EarlyStopping(patience=5,monitor='val_accuracy', mode='max', min_delta=1), 
             keras.callbacks.ModelCheckpoint('ResNet152_Weights_ES.h5',
              verbose=1, save_best_only=True),
             keras.callbacks.TensorBoard(log_dir='logs',
                update_freq='epoch',
                histogram_freq=0,
                write_graph=True,
                write_images=False)]
# Fitting (Training)             
history = model.fit_generator(generator=train_generator,
    validation_data=val_generator,epochs=100, 
    steps_per_epoch=len(os.listdir(train_images_path)) // batch_size, 
    validation_steps =len(os.listdir(val_images_path))// batch_size,
    verbose=1,
    callbacks = Callbacks)

"""## Save Results"""

# #==============
# # Save History
# #==============
np.save('ResNet152_history_ES.npy',history.history)
# #================
# # Save Model
# #================
model.save('ResNet152_Model.h5')
# Delete Model
# del model
# del history

#==================================================
"""### Load Model"""
#===================
model = keras.models.load_model("ResNet152_Weights_ES.h5",compile=False)
loss = sm.losses.cce_dice_loss
model.compile( 'Adam',
    loss=loss,
    metrics=[sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)])
#==============
# Load History
#==============
history=np.load('ResNet152_history_ES.npy',allow_pickle='TRUE').item()

"""## Evaluate The Model"""

#================
# Evaluate Model
#================
score = model.evaluate(test_generator_eval)
print('Test loss:', score[0])
print('Test accuracy', score[1])

"""## Visualize the Metrics"""

#=================================
"""### Visualize fitting curve""" 
#================================= 
plt.figure(figsize=(30, 5), num = 'Metrics')
plt.subplot(121)
plt.plot(history['iou_score'])
plt.plot(history['val_iou_score'])
plt.title('Model iou_score')
plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
"""## Prediction"""

#================
# Predict Model
#================
test_images, test_labels = next(iter(test_generator_eval))
prediction = model.predict(test_generator)
#-----------------------------------------
# Thresholding
prediction[prediction>=0.5]=1
prediction[prediction<0.5]=0
# RGB
pred = np.squeeze(prediction)

masks = np.squeeze(test_labels)

images = np.squeeze(test_images)

"""## Visualize Prediction"""

#===================================
"""### Visualize the Results"""
#===================================
print('pred shape: ',pred.shape,'masks.shape: ',masks.shape,'images.shape: ',images.shape,'\n' )
plt.figure(num='ResNet_152')

plt.subplot('221')
plt.imshow(np.squeeze(images[1,:,:]),cmap='gray')
plt.title('Train Image')
plt.axis('off')

plt.subplot('222')
plt.imshow(masks[1,:,:,1:4])
plt.title('Train Mask')
plt.axis('off')

plt.subplot('223')
plt.imshow(np.squeeze(images[1,:,:]),cmap='gray')
plt.title('Test Image')
plt.axis('off')

plt.subplot('224')
plt.imshow(pred[1,:,:,1:4])
plt.title('Prediction')
plt.axis('off')

plt.show()