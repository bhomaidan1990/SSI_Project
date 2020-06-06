import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import keras
import numpy as np

def KerasPredict(image2D, modelPath='Trained_model/Basic_unet/unet_basic_cce.h5'):
    # Image
    image3D= np.expand_dims(image2D, axis=0)
    imageTensor = np.expand_dims(image3D, axis=3)

    # optim = keras.optimizers.Adam(lr = 1e-5, beta_1=0.9, beta_2=0.999,)
    # metrics = [keras.metrics.MeanIoU]

    if(os.path.isfile(str(modelPath))):
        # # Load Pretrained Model
        # model = keras.models.load_model(modelPath,compile=False)
       
        # # Compile The PreTrained Model
        # model.compile(loss=loss, optimizer=optim, metrics=metrics)

        model = keras.models.load_model(modelPath)  

        prediction = model.predict(imageTensor,verbose=1)

        # Threshold the output
        prediction[prediction>=0.5] = 1
        prediction[prediction< 0.5] = 0

        # Squeeze the Extra dimensions
        pred = np.squeeze(prediction)

    else:
        pred = None
   
    return pred
#====================================================

def PyTorchPredict(image2D,modelPath='Trained_model/mobilenet_model.tar'):

    pass
#  return pred # (256x256x4) 0 or 1 values