import torch
import torchvision
from torchModel import UNet, mobileUnet
from torchDataloader import crop3D
import segmentation_models_pytorch as smp
import numpy as np
from skimage.color import gray2rgb
"""
# This Function to be called `PyTorchPredict` with image input (HxW) 2D,
# and output: 256x256x4 (0 and 1 values only, no probabilites here)
# to be added to the script "SSI_Project/Project_Code/Predict.py"
# PyTorchPredict(image2D, backbone, modelPath) ---> 256x256x4 prediction 
"""
def eval(img, backbone='mobilenet', model_file='./model.tar'):
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
    ip_size = img.shape
    img = gray2rgb(img)
    img = np.transpose(2,0,1)
    img = np.expand_dims(img, 0)

    if backbone == 'mobilenet':
        net = mobileUnet(torchvision.models.mobilenet_v2())
        img = crop3D(img, (224,224))
    elif backbone == 'efficientnet':
        net = smp.Unet('efficientnet-b7', classes = 4)
    else:
        net = UNet(num_classes=4, input_channel=3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        torch.cuda.empty_cache() 
    net.to(device)
    net.eval()
    img_torch = torch.from_numpy(img).type(torch.FloatTensor).to(device)
    # img_torch = img_torch
    pretrained = torch.load(model_file)
    net.load_state_dict(pretrained['model'])

    pred = net(img_torch)
    pred = pred.cpu().detach().numpy()

    pred[pred > 0.5] = 1
    pred[pred != 1] = 0
    if backbone == 'mobilenet':
        pred = np.reshape(4,224,224)
    else:
        pred = np.reshape(4,ip_size[0], ip_size[1])

    pred = np.transpose(pred, [1,2,0])
    return pred