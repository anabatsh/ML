
import numpy as np

from skimage.io import imread
from skimage.util import invert, img_as_float
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation, binary_erosion, area_opening, square, disk, diamond
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.color import label2rgb, rgb2gray
from skimage.transform import rotate

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import glob
import natsort
import os
import cv2

from math import ceil, sqrt
from collections.abc import Iterable
import matplotlib.pyplot as plt

def _make_mask_a(img):

    # перевод в полутоновое изображение
    img = rgb2gray(img)

    # инвертирование
    img = invert(img)

    # бинаризация по Оцу
    thresh = threshold_otsu(img)
    mask = img > thresh

    # заполнение точек внутри
    mask = binary_dilation(mask, square(7))
    mask = binary_erosion(mask, square(7))

    # отбор фигур площадью меньше 2500 (примерная площадь триминошки)
    mask = area_opening(mask, 2500)

    # очистка внешней границы от влияния градиента освещенности
    mask = clear_border(mask)

    # уменьшение перемычек для разделения фигур
    mask_new = binary_erosion(mask, square(19))
    mask_new = area_opening(mask_new, 500)

    # перебор оставшихся объектов на соответствие своей выпуклой оболочке
    label_image = label(mask_new)
    image_label_overlay = label2rgb(label_image, image=mask_new, bg_label=0)

    for region in regionprops(label_image):

        minr, minc, maxr, maxc = region.bbox

        sub_img = img_as_float(region.image)
        sub_convex = img_as_float(region.convex_image)

        sub_diff = (sub_convex-sub_img).sum()
        if (sub_diff > 150):
            label_image[minr:maxr, minc:maxc][sub_img==1.0] = 0

    # возврат к исходным размерам фигур
    label_image_new = binary_dilation(label_image, disk(19))

    # генерация маски
    mask = img_as_float(mask)
    mask = label_image_new*mask

    return mask
    
def _make_mask_b(img):

	img_new = img.copy() / 255
	B = img_new[...,2]

	B[B > 0.23] = 0
	B = B > 0
	B = area_opening(B, 100)
	B = binary_dilation(B, disk(5))
	B = binary_erosion(B, disk(5))

	B_new = binary_erosion(B, disk(13))
	B_new = area_opening(B_new, 800)

	# перебор оставшихся объектов на соответствие своей выпуклой оболочке
	label_image = label(B_new)

	for region in regionprops(label_image):

	    minr, minc, maxr, maxc = region.bbox

	    sub_img = img_as_float(region.image)
	    sub_convex = img_as_float(region.convex_image)

	    sub_diff = (sub_convex-sub_img).sum()
	    if (sub_diff > 250):
	    	label_image[minr:maxr, minc:maxc][sub_img==1.0] = 0
	    
	# возврат к исходным размерам фигур
	label_image_new = binary_dilation(label_image, diamond(21))

	mask = img_as_float(label_image_new)
	mask = binary_dilation(mask, disk(5))
	mask = binary_erosion(mask, disk(5))
	mask = B*mask

	return mask


def _generate_dataset(img, mask, s, n):

    dataset = []
    h, w, c = img.shape
    h, w = h - s, w - s

    for i in range(n):
        y = np.random.randint(170, h-100)
        x = np.random.randint(170, w-120)

        for angle in (0, 90, 180, 270):
            sub_img  = rotate(img[y:y+s, x:x+s,...], angle, preserve_range=True)
            sub_mask = rotate(mask[y:y+s, x:x+s,...], angle, preserve_range=True)

            for b in np.arange(0.2, 1.2, step=0.2):

                dataset.append({'img': sub_img*b, 'mask': sub_mask})
                dataset.append({'img': invert(sub_img*b), 'mask': sub_mask})

    return dataset


def simple_download(dir):
    return [imread(file) for file in glob.glob(dir+'/*')]


class GenerateDataset(Dataset):
    def __init__(self, base_imgs, transforms=None):

        base_masks = self.make_masks(base_imgs)

        self.dataset = self.generate_dataset(base_imgs, base_masks)
        self.transforms = transforms

    def make_masks(self, base_imgs):
    	mask_a = _make_mask_a(base_imgs[0])
    	mask_b = _make_mask_b(base_imgs[1])
    	return [mask_a, mask_b]

    def generate_dataset(self, base_imgs, base_masks):
        dataset = []
        for img, mask in zip(base_imgs, base_masks):
            dataset = dataset + _generate_dataset(img, mask, 200, 10)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

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

        img = torch.from_numpy(img.astype(np.float32).transpose((2, 0, 1))).contiguous()
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


def show_images(images, labels=None, title=None, transform=None, figsize=(12, 12), use_BGR=False):
    fig = plt.figure(figsize=figsize, linewidth=5, edgecolor="#04253a")
    grid_val = ceil(sqrt(len(images)))
    grid_specs = plt.GridSpec(grid_val, grid_val)

    for i, image in enumerate(images):
        ax = fig.add_subplot(grid_specs[i // grid_val, i % grid_val])
        ax.axis('off')

        if transform is not None:
            image = transform(image)

        if labels is not None:
            ax_title = labels[i]
        else:
            ax_title = '#{}'.format(i+1)

        ax.set_title(ax_title)
        if use_BGR and image.shape[-1] == 3:
            image = image[..., ::-1]

        ax.imshow(image.squeeze(), cmap='gray')

    if title:
        fig.suptitle(title, y=0.93, fontsize='xx-large')
    plt.show()


