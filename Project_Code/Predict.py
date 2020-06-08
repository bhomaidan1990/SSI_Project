import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import keras
import tensorflow as tf
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchModel import UNet, mobileUnet
from torchDataloader import crop3D, oneHotMask
import segmentation_models_pytorch as smp
from skimage.color import gray2rgb
from numba import cuda

def KerasPredict(image2D, modelPath='Trained_model/Modified_Unet/final_unet_cce.h5'):

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
       
        del model
        cuda.select_device(0)
        cuda.close()

        # Squeeze the Extra dimensions
        pred = np.squeeze(prediction)

    else:

        pred = None
   
    return pred
#====================================================

def eval(img, backbone='mobilenet', model_file='Trained_model/mobileNet_new.tar'):
    """

    @param img: input image, with order at [N,C,H,W], N for batch size, C for channel.
                for all these models, input size should be [N, 3, 256, 256]
    @param backbone: choose the decoder, 'mobilenet' or 'efficientnet' or 'unet'
    @param model_file: file contains the pretrained weights. For now, the model only runs on GPU
    @return:
        pred: prediction of the input, 4D numpy array, with dimension order [N,C,H,W]
              for mobileNet, dimension for prediction would be [N, 4, 224, 224]
              for others, output dimensions would be [N, 4, 256, 256]
    """
    
    img = gray2rgb(img)
    img = np.transpose(img, (2,0,1))
    img = np.expand_dims(img, 0)

    if backbone == 'mobilenet':
        net = mobileUnet(torchvision.models.mobilenet_v2())
        img = crop3D(img, (224,224))
    elif backbone == 'efficientnet':
        net = smp.Unet('efficientnet-b7', classes = 4)
    else:
        net = smp.UNet(num_classes=4, input_channel=3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.empty_cache() 
    else:
        print('This model required GPU')
        return
    net.to(device)
    net.eval()
    img_torch = torch.from_numpy(img).type(torch.FloatTensor).to(device)
    # img_torch = img_torch
    pretrained = torch.load(model_file)
    net.load_state_dict(pretrained['model'])
    
    pred = net(img_torch)

    pred = F.softmax(pred, dim=1)

    pred = pred.cpu().detach().numpy()

    # # Argmax
    # #============================
    # pred = np.argmax(pred,axis=1)

    # [_, x,y] = pred.shape
    
    # pred = np.reshape(pred, (x,y))   
    # pred = oneHotMask(pred)


    # Thresholding 
    # #============================
    pred[pred>=0.5] = 1
    pred[pred< 0.5] = 0 

    pred = np.squeeze(pred)

    pred = np.transpose(pred, [1,2,0])
    #=============================
    del net
    if device.type == 'cuda':
        torch.cuda.empty_cache() 

    return pred


def crop2D_mask(img, newSize):
    x = img.shape[0]
    y = img.shape[1]
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
    if(len(img.shape)==3):
        return img[cx:x-cx-ex,cy:y-cy-ey, :]
    if(len(img.shape)==2):
        return img[cx:x-cx-ex,cy:y-cy-ey]