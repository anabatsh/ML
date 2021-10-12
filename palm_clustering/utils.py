import numpy as np

from skimage.io import imread, imsave
from skimage.util import img_as_float, img_as_ubyte
from skimage.morphology import area_opening, convex_hull_image
from skimage.segmentation import clear_border
from skimage.color import rgb2gray
from skimage.transform import AffineTransform, warp, rotate, resize

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import glob
import os
import natsort

import joblib
from joblib import Parallel, delayed


def _download(file):
    img = imread(file)
    img = resize(img, (704, 512))
    return img

def _name(file):
    return int(file[-7:-4])

def download(dir):
    images_names = []
    images = []
    for file in glob.glob(dir+'/*.tif'):
        images_names.append(_name(file))
        images.append(_download(file))
    return images, images_names


def _resize(img):
    img = resize(img, (176, 128), preserve_range=True)
    return img

def _deresize(img):
    img = resize(img, (704, 512), preserve_range=True)
    return img
    
def _make_mask(img, area=20):
    img = rgb2gray(img)
    palm = img > img.mean()

    thresh = np.quantile(img, 0.8)
    bin_img = (img > thresh)
    bin_img = clear_border(bin_img)
    bin_img = area_opening(bin_img, area)

    bin_img = convex_hull_image(bin_img)
    bin_img = bin_img * palm

    return bin_img


def _img_shift(img, shift):
    transform = AffineTransform(translation=shift)
    shifted_img = warp(img, transform, preserve_range=True)
    shifted_img = shifted_img.astype(img.dtype)
    return shifted_img

def _angle_grid(a_min, a_max, a_step):
    a = np.arange(a_min, a_max + a_step, step=a_step)
    return a

def _shift_grid(s_min, s_max, s_step):
    s = np.arange(s_min, s_max + s_step, step=s_step)
    return s

def _brightness_grid(b_min, b_max, b_step):
    b = np.arange(b_min, b_max + b_step, step=b_step)
    return b

def _overall_grid(*grids):
    m = np.meshgrid(*grids)
    m = np.stack(m, axis=-1).reshape(-1, 4)
    return m

def _generate_params():
    # 1 img -> 400 imgs
    a_grid = _angle_grid(-180, 135, 45) # 8
    sx_grid = _shift_grid(0, 25, 5) # 5
    sy_grid = _shift_grid(0, 25, 5) # 5
    b_grid = _brightness_grid(0.5, 1., 0.5) # 2
    params = _overall_grid(a_grid, sy_grid, sx_grid, b_grid)
    return params

def _make_save_sample(img, mask, param, img_dir, mask_dir, ind):

    angle, x_shift, y_shift, brightness = param
    img_new  = rotate(img, angle, preserve_range=True, mode='edge')
    img_new  = _img_shift(img_new, (y_shift, x_shift))
    img_new  = img_new * brightness


    mask_new = rotate(mask, angle, preserve_range=True)
    mask_new = _img_shift(mask_new, (y_shift, x_shift))

    imsave(img_dir + f'/{ind}.png', img_as_ubyte(img_new))
    imsave(mask_dir + f'/{ind}.png', img_as_ubyte(mask_new))
    

def generate_train_dataset(base_imgs, img_dir, mask_dir):

    base_imgs = [_resize(img) for img in base_imgs]
    base_masks = [_make_mask(img) for img in base_imgs]
    params = _generate_params()
    p_len = len(params)
    bi_len = len(base_imgs)

    for i in range(bi_len):

        num_of_cpu = joblib.cpu_count()

        delayed_funcs = [delayed(_make_save_sample)(base_imgs[i],
                                                    base_masks[i],
                                                    param,
                                                    img_dir,
                                                    mask_dir, i*len(params)+j) for j, param in enumerate(params)]

        parallel_pool = Parallel(n_jobs=num_of_cpu)
        parallel_pool(delayed_funcs)
        
        
class MyDataset(Dataset):
    def __init__(self, img_dir, mask_dir=None, transforms=None):

        self.transforms = transforms

        self.imgs = natsort.natsorted(glob.glob(img_dir + '/*'))       
        self.masks = natsort.natsorted(glob.glob(mask_dir + '/*')) 

    def __len__(self):
        return len(self.imgs)
                   
    def __getitem__(self, idx):
        
        img = imread(self.imgs[idx])
        mask = imread(self.masks[idx])

        sample = {'img': img, 'mask': mask}

        if self.transforms:
            sample = self.transforms(sample)

        return sample
    
    
class Compose():

    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class ToTensor():

    def to_tensor(self, img):
        if not isinstance(img, np.ndarray):
            img = np.asarray(img)

        if img.ndim == 2:
            img = img[:, :, None]

        img = torch.from_numpy((img/255).astype(np.float32).transpose((2, 0, 1))).contiguous()
        return img

    def __call__(self, sample):
        if isinstance(sample, np.ndarray):
            return  self.to_tensor(sample)

        if isinstance(sample, dict):
            img, mask = sample['img'], sample['mask']
            img = self.to_tensor(img)
            mask = self.to_tensor(mask)
            return {'img': img, 'mask': mask}


class Normalize():

    def normalize(self, img):
        max = torch.max(img)
        min = torch.min(img)
        mean = (max - min) / 2
        std = max - mean

        img = (img - mean) / std
        return img

    def __call__(self, sample):
        if isinstance(sample, torch.Tensor):
            return  self.normalize(sample)

        if isinstance(sample, dict):
            img, mask = sample['img'], sample['mask']
            img = self.normalize(img)
            return {'img': img, 'mask': mask}
        
        
        

