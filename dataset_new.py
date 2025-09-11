import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import random
import torchvision.transforms.functional as F
from glob import glob
import torchvision.transforms as transforms
from PIL import ImageEnhance

def cv_random_flip(img, label):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label

def randomCrop(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)

def randomRotation(image, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return image, label


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


class CustomDataset(Dataset):
    def __init__(self, data_path , transform = None, mode = 'Training',size=512, fold=0, syn_path=None):
        random.seed(2025)
        print("loading data from the directory :",data_path)
        path=data_path
        if 'COVID' in data_path or 'covid' in data_path:
            images = sorted(glob(os.path.join(path, "images/*.jpg")))
            masks = sorted(glob(os.path.join(path, "masks/*.png")))
            paired_data = list(zip(images, masks))
            random.shuffle(paired_data)
            images, masks = zip(*paired_data)
        else:
            images = sorted(glob(os.path.join(path, "images/*.png")))
            masks = sorted(glob(os.path.join(path, "masks/*.png")))
            paired_data = list(zip(images, masks))
            random.shuffle(paired_data)
            images, masks = zip(*paired_data)

        fold_size = len(images) // 4
        if mode == 'Training':
            if fold != 3:
                val_image = images[fold*fold_size:(fold+1)*fold_size]
                val_label = masks[fold*fold_size:(fold+1)*fold_size]
            else:
                val_image = images[fold*fold_size:]
                val_label = masks[fold*fold_size:]
            self.name_list = list(filter(lambda x: x not in val_image, images))
            self.label_list = list(filter(lambda x: x not in val_label, masks))
        else:
            if fold != 3:
                self.name_list = images[fold*fold_size:(fold+1)*fold_size]
                self.label_list = masks[fold*fold_size:(fold+1)*fold_size]
            else:
                self.name_list = images[fold*fold_size:]
                self.label_list = masks[fold*fold_size:]
        if syn_path is not None and mode == 'Training':
            syn_images = sorted(glob(os.path.join(syn_path, "images/*.png")))
            syn_masks = sorted(glob(os.path.join(syn_path, "masks/*.png")))
            self.name_list = self.name_list + syn_images
            self.label_list = self.label_list + syn_masks
        print(fold, mode, len(self.name_list), len(self.label_list))
        self.data_path = path
        self.mode = mode
        self.size = size

        self.transform = transform
        if mode != 'Training':
            transform_list = [transforms.ToTensor(), ]
            transform_mask = transforms.Compose(transform_list)
            self.transform_mask = transform_mask

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(mask_name)

        img = Image.open(img_path).convert('RGB')
        if self.mode == 'Training':
            mask = Image.open(msk_path).convert('L')#.resize((self.size,self.size),resample=Image.Resampling.NEAREST)
            img, mask = cv_random_flip(img, mask)
            img, mask = randomCrop(img, mask)
            img, mask = randomRotation(img, mask)
            img = colorEnhance(img)
            mask = mask.resize((self.size, self.size),resample=Image.Resampling.NEAREST)
        else:
            mask = Image.open(msk_path)#.convert('L').resize((self.size,self.size),resample=Image.Resampling.NEAREST)

        if self.transform:
            img = self.transform(img)
            if self.mode == 'Training':
                mask = self.transform(mask)
            else:
                mask = self.transform_mask(mask)

        if self.mode == 'Training':
            return (img, mask)
        else:
            return (img, mask, name)
