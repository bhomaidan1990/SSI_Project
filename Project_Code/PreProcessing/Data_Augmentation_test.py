#================================
#===  @Author: Belal Hmedan  ====
#================================
# import necessary libraries
import os
import glob
import cv2
import numpy as np # linear algebra
from skimage.io import imread, imsave, imshow
import matplotlib.pyplot as plt
from skimage.util import montage
from keras.preprocessing.image import ImageDataGenerator
#=================================================
# Input Path
train_image_dir = 'Dataset'    # change dir and subdirs up to your files
# Train file name
img_fname       = 'images'  # folder_name train images
mask_fname      = 'masks'  # folder_name of train masks
Augmenting_index = 1 # chane the index whenever you change augmenting parameters down to generate new images.
# Output Path
out_path = 'Dataset' # state the path you need to extract augmented images
output_img_size = 128       # change your output images size
#-------------------------------------------------
# Augmentation Parameters
IMG_SCALING = (1, 1)
dg_args = dict(featurewise_center = False, 
                  samplewise_center = False,
                  rotation_range = 5, 
                  width_shift_range = 0.05, 
                  height_shift_range = 0.05, 
                  shear_range = 0.05,  
                  horizontal_flip = True, 
                  vertical_flip = True,
                  fill_mode = 'nearest',
                   data_format = 'channels_last')
#=================================================
BATCH_SIZE = len(os.listdir(os.path.join(train_image_dir,img_fname)))
#=================================================
montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
#=================================================
def get_all_imgs():
    img_path = os.path.join(train_image_dir,img_fname)
    images = glob.glob(os.path.join(img_path,'*.*'))
    mask_path = os.path.join(train_image_dir,mask_fname)
    masks = glob.glob(os.path.join(mask_path,'*.*'))
    return [os.path.basename(image) for image in images], [os.path.basename(mask) for mask in masks]

# print(get_all_imgs())
TRAIN_IMGS=get_all_imgs()
# Image and mask should have same, but different folder
def make_image_gen(img_file_list=TRAIN_IMGS, batch_size = BATCH_SIZE):
    all_batches = TRAIN_IMGS
    out_img = []
    out_mask = []
    img_path  = os.path.join(train_image_dir,img_fname)
    mask_path = os.path.join(train_image_dir,mask_fname)
    while True:
        # np.random.shuffle(all_batches)
        for num in range(len(all_batches[0])):
            c_img  = imread(os.path.join(img_path,all_batches[0][num]))
            c_mask = imread(os.path.join(mask_path,all_batches[1][num]))
            if IMG_SCALING is not None:
                # c_img = cv2.resize(c_img,(output_img_size,output_img_size),interpolation = cv2.INTER_AREA)
                o_img = c_img.copy()
                # c_mask = cv2.resize(c_mask,(output_img_size,output_img_size),interpolation = cv2.INTER_AREA)
                o_mask = c_mask.copy()
            c_mask = np.reshape(c_mask,(c_mask.shape[0],c_mask.shape[1],-1))
            c_mask = c_mask > 0
            out_img += [c_img]
            out_mask += [c_mask]
            if len(out_img)>=batch_size:
                yield np.stack(out_img, 0)/255.0, np.stack(out_mask, 0)
                out_img, out_mask=[], []

def write_images(img_list,mask_list, out_path):
    img_path = os.path.join(out_path,'images')
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    for i,image in enumerate(img_list):
        imsave(os.path.join(img_path,'img_%d_%d.png'%(i,Augmenting_index)), image)
    mask_path = os.path.join(out_path,'masks')
    if not os.path.exists(mask_path):
        os.mkdir(mask_path)
    for i,mask in enumerate(mask_list):
        imsave(os.path.join(mask_path,'img_%d_%d.png'%(i,Augmenting_index)), mask)
#------------------------------------------------------------------------

#========================================================================
image_gen = ImageDataGenerator(**dg_args)
label_gen = ImageDataGenerator(**dg_args)

def create_aug_gen(in_gen, seed = None):
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(np.expand_dims(255*in_x,axis=3), batch_size = in_x.shape[0], seed = seed, shuffle=True)
        g_y = label_gen.flow(in_y, batch_size = in_x.shape[0],seed = seed, shuffle=True)
        yield next(g_x)/255.0, next(g_y)
#-----------------------------------------------------------------------        
train_gen = make_image_gen()
cur_gen = create_aug_gen(train_gen)
t_x, t_y = next(cur_gen)
print('x', t_x.shape, t_x.dtype, t_x.min(), t_x.max())
print('y', t_y.shape, t_y.dtype, t_y.min(), t_y.max())

write_images(t_x, t_y, out_path)

# only keep first 9 samples to examine in detail
# t_x = t_x[:9]
# t_y = t_y[:9]

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 10))
# ax1.imshow(montage(t_x[:, :, :, 0]), cmap='gray')
# ax1.set_title('images')
# ax2.imshow(montage(t_y[:, :, :, 0]), cmap='gray')
# ax2.set_title('masks')
# plt.axis('off')
# plt.show()
# plt.close('all')