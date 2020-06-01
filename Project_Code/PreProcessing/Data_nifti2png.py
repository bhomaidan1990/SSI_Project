#================================
#===  @Author: Belal Hmedan  ====
#================================
# import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt 
import glob
import nibabel as nib
from skimage.io import imsave, imread
from skimage.transform import resize
#-------------------------------------------
# Normalization
def normalize_zscore(data, z=2, offset=0.5, clip=False):
    """
    Normalize contrast across volume
    """
    mean = np.mean(data)
    std = np.std(data)
    img = ((data - mean) / (2 * std * z) + offset) 
    if clip:
        # print ('Before')
        # print (np.min(img), np.max(img))
        img = np.clip(img, -0.0, 1.0)
        # print ('After clip')
        # print (np.min(img), np.max(img))
    return img
#-------------------------------------------
def normalize_minmax(data):
    """
    Normalize contrast across volume
    """
    _min = np.float(np.min(data))
    _max = np.float(np.max(data))
    if (_max-_min)!=0:
        img = (data - _min) / (_max-_min)
    else:
        img = np.zeros_like(data)            
    return img
#-------------------------------------------
# Make Folders for Output
if (not os.path.exists('Dataset')):
    os.mkdir('Dataset')
    os.mkdir('Dataset/images')
    os.mkdir('Dataset/masks')
    os.mkdir('Dataset/masks/Lung_masks')
    os.mkdir('Dataset/masks/Covid_masks')
    os.mkdir('Dataset/masks/Lung_and_Covid_masks')
#-------------------------------------------
# Images
path = 'COVID-19-CT-Seg_20cases'
Dataset = glob.glob( os.path.join(path, '*.gz') )
ctr = 0
for image in Dataset:
    # Load images voxel
    images = nib.load(image).get_fdata()
    # Normalize data
    images = normalize_zscore(images)
    # Save as PNG
    ctr+=1
    if(not os.path.exists('Dataset/images/Case_'+str(ctr))):
        os.mkdir('Dataset/images/Case_'+str(ctr))
    for _id in range(images.shape[2]):
        imsave(os.path.join('Dataset/images','Case_'+
            str(ctr),str(ctr)+'_'+str(_id+1)+'.png'),
             resize(images[:,:,_id],(256,256),anti_aliasing=True).astype(np.uint8),check_contrast=False)
#-------------------------------------------
# Lung Masks
path = 'Lung_Mask'
Dataset = glob.glob( os.path.join(path, '*.gz') )
ctr = 0
for image in Dataset:
    # Load masks voxel
    images = nib.load(image).get_fdata()
    print('Lungs before: ',np.unique(images))
    # Save it as PNG
    ctr+=1
    if(not os.path.exists('Dataset/masks/Lung_masks/Case_'+str(ctr))):
        os.mkdir('Dataset/masks/Lung_masks/Case_'+str(ctr))
    for _id in range(images.shape[2]):
        imsave(os.path.join('Dataset/masks/Lung_masks','Case_'+
            str(ctr),str(ctr)+'_'+str(_id+1)+'.png'),
             resize(images[:,:,_id],(256,256),preserve_range=True).astype(np.uint8),check_contrast=False)
#-------------------------------------------
# Covid-19 Masks
path = 'Infection_Mask'
Dataset = glob.glob( os.path.join(path, '*.gz') )
ctr = 0
for image in Dataset:
    # Load masks voxel
    images = nib.load(image).get_fdata()
    # Save it as PNG
    ctr+=1
    if(not os.path.exists('Dataset/masks/Covid_masks/Case_'+str(ctr))):
        os.mkdir('Dataset/masks/Covid_masks/Case_'+str(ctr))
    for _id in range(images.shape[2]):
        imsave(os.path.join('Dataset/masks/Covid_masks','Case_'+
            str(ctr),str(ctr)+'_'+str(_id+1)+'.png'),
             resize(images[:,:,_id],(256,256),preserve_range=True).astype(np.uint8),check_contrast=False)
#-------------------------------------------
# Lungs + Covid-19 Masks
path = 'Lung_and_Infection_Mask'
Dataset = glob.glob( os.path.join(path, '*.gz') )
ctr = 0
for image in Dataset:
    # Load masks voxel
    images = nib.load(image).get_fdata()
    print('before: ',np.unique(images))
    # Save it as PNG
    ctr+=1
    if(not os.path.exists('Dataset/masks/Lung_and_Covid_masks/Case_'+str(ctr))):
        os.mkdir('Dataset/masks/Lung_and_Covid_masks/Case_'+str(ctr))
    for _id in range(images.shape[2]):
        resized = resize(images[:,:,_id],(256,256),preserve_range=True).astype(np.uint8)
        imsave(os.path.join('Dataset/masks/Lung_and_Covid_masks','Case_'+
            str(ctr),str(ctr)+'_'+str(_id+1)+'.png'),
             resized,check_contrast=False)
#-------------------------------------------