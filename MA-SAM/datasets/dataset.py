import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from einops import repeat
from icecream import ic
import pandas as pd
import pickle
import PIL.Image
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
import cv2

HU_min, HU_max = -200, 250
data_mean = 50.21997497685108
data_std = 68.47153712416372

def read_image(path):
    with open(path, 'rb') as file:
        img = pickle.load(file)
        return img

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(0, 1))
    label = np.rot90(label, k, axes=(0, 1))
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-15, 15)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def convert_to_PIL(img: np.array) -> PIL.Image:
    '''
    img should be normalized between 0 and 1
    '''
    img = np.clip(img, 0, 1)
    return PIL.Image.fromarray((img * 255).astype(np.uint8))

def convert_to_PIL_label(label):
    return PIL.Image.fromarray(label.astype(np.uint8))

def convert_to_np(img: PIL.Image) -> np.array:
    return np.array(img).astype(np.float32) / 255

def convert_to_np_label(label):
    return np.array(label).astype(np.float32)

def random_erasing(
    imgs,
    label,
    scale_z=(0.02, 0.33),
    scale=(0.02, 0.05),
    ratio=(0.3, 3.3),
    apply_all: int = 0,
    rng: np.random.Generator = np.random.default_rng(0),
):

    # determine the box
    imgshape = imgs.shape
    
    # nx and ny
    while True:
        se = rng.uniform(scale[0], scale[1]) * imgshape[0] * imgshape[1]
        re = rng.uniform(ratio[0], ratio[1])
        nx = int(np.sqrt(se * re))
        ny = int(np.sqrt(se / re))
        if nx < imgshape[1] and ny < imgshape[0]:
            break

    # determine the position of the box
    sy = rng.integers(0, imgshape[0] - ny + 1)
    sx = rng.integers(0, imgshape[1] - nx + 1)

    # print(nz, ny, nx, sz, sy, sx)
    filling = rng.uniform(0, 1, size=[ny, nx])
    filling = filling[:,:,np.newaxis]
    filling = np.repeat(filling, imgshape[-1], axis=-1)

    # erase
    imgs[sy:sy + ny, sx:sx + nx, :] = filling
    label[sy:sy + ny, sx:sx + nx, :] = 0.

    return imgs, label

def posterize(img, label, v):
    '''
    4 < v < 8
    '''
    v = int(v)
    
    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = PIL.ImageOps.posterize(img_curr, bits=v)
        img_curr = convert_to_np(img_curr)
        img[:,:,slice_indx] = img_curr

    return img, label

def contrast(img, label, v):
    '''
    0.1 < v < 1.9
    '''
    
    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = PIL.ImageEnhance.Contrast(img_curr).enhance(v)
        img_curr = convert_to_np(img_curr)
        img[:,:,slice_indx] = img_curr

    return img, label

def brightness(img, label, v):
    '''
    0.1 < v < 1.9
    '''
    
    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = PIL.ImageEnhance.Brightness(img_curr).enhance(v)
        img_curr = convert_to_np(img_curr)
        img[:,:,slice_indx] = img_curr

    return img, label

def sharpness(img, label, v):
    '''
    0.1 < v < 1.9
    '''
    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = PIL.ImageEnhance.Sharpness(img_curr).enhance(v)
        img_curr = convert_to_np(img_curr)
        img[:,:,slice_indx] = img_curr

    return img, label

def identity(img, label, v):
    return img, label

def adjust_light(image, label):
    image = image*255.0
    gamma = random.random() * 3 + 0.5
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    for slice_indx in range(image.shape[2]):
        img_curr = image[:,:,slice_indx]
        img_curr = cv2.LUT(np.array(img_curr).astype(np.uint8), table).astype(np.uint8)
        image[:,:,slice_indx] = img_curr
    image = image/255.0

    return image, label

def shear_x(img, label, v):
    '''
    -0.3 < v < 0.3
    '''
    shear_mat = [1, v, -v * img.shape[1] / 2, 0, 1, 0]  # center the transform
    
    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = img_curr.transform(img_curr.size, PIL.Image.AFFINE, shear_mat, resample=PIL.Image.BILINEAR)
        img_curr = convert_to_np(img_curr)
        img[:,:,slice_indx] = img_curr
        # print(img.shape)

    for slice_indx in range(label.shape[2]):
        label_curr = label[:,:,slice_indx]
        label_curr = convert_to_PIL_label(label_curr)
        label_curr = label_curr.transform(label_curr.size, PIL.Image.AFFINE, shear_mat, resample=PIL.Image.NEAREST)
        label_curr = convert_to_np_label(label_curr)
        label[:,:,slice_indx] = label_curr
        # print(label.shape)

    return img, label

def shear_y(img, label, v):
    '''
    -0.3 < v < 0.3
    '''
    shear_mat = [1, 0, 0, v, 1, -v * img.shape[0] / 2]  # center the transform

    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = img_curr.transform(img_curr.size, PIL.Image.AFFINE, shear_mat, resample=PIL.Image.BILINEAR)
        img_curr = convert_to_np(img_curr)
        img[:,:,slice_indx] = img_curr
        # print(img.shape)

    for slice_indx in range(label.shape[2]):
        label_curr = label[:,:,slice_indx]
        label_curr = convert_to_PIL_label(label_curr)
        label_curr = label_curr.transform(label_curr.size, PIL.Image.AFFINE, shear_mat, resample=PIL.Image.NEAREST)
        label_curr = convert_to_np_label(label_curr)
        label[:,:,slice_indx] = label_curr
        # print(label.shape)

    return img, label

def translate_x(img, label, v):
    '''
    -0.45 < v < 0.45
    '''
    translate_mat = [1, 0, v * img.shape[1], 0, 1, 0]

    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = img_curr.transform(img_curr.size, PIL.Image.AFFINE, translate_mat, resample=PIL.Image.BILINEAR)
        img_curr = convert_to_np(img_curr)
        img[:,:,slice_indx] = img_curr

    for slice_indx in range(label.shape[2]):
        label_curr = label[:,:,slice_indx]
        label_curr = convert_to_PIL_label(label_curr)
        label_curr = label_curr.transform(label_curr.size, PIL.Image.AFFINE, translate_mat, resample=PIL.Image.NEAREST)
        label_curr = convert_to_np_label(label_curr)
        label[:,:,slice_indx] = label_curr

    return img, label

def translate_y(img, label, v):
    '''
    -0.45 < v < 0.45
    '''
    translate_mat = [1, 0, 0, 0, 1, v * img.shape[0]]

    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = img_curr.transform(img_curr.size, PIL.Image.AFFINE, translate_mat, resample=PIL.Image.BILINEAR)
        img_curr = convert_to_np(img_curr)
        img[:,:,slice_indx] = img_curr

    for slice_indx in range(label.shape[2]):
        label_curr = label[:,:,slice_indx]
        label_curr = convert_to_PIL_label(label_curr)
        label_curr = label_curr.transform(label_curr.size, PIL.Image.AFFINE, translate_mat, resample=PIL.Image.NEAREST)
        label_curr = convert_to_np_label(label_curr)
        label[:,:,slice_indx] = label_curr

    return img, label

def scale(img, label, v):
    '''
    0.6 < v < 1.4
    '''
    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = img_curr.transform(img_curr.size, PIL.Image.AFFINE, [v, 0, 0, 0, v, 0], resample=PIL.Image.BILINEAR)
        img_curr = convert_to_np(img_curr)
        img[:,:,slice_indx] = img_curr

    for slice_indx in range(label.shape[2]):
        label_curr = label[:,:,slice_indx]
        label_curr = convert_to_PIL_label(label_curr)
        label_curr = label_curr.transform(label_curr.size, PIL.Image.AFFINE, [v, 0, 0, 0, v, 0], resample=PIL.Image.NEAREST)
        label_curr = convert_to_np_label(label_curr)
        label[:,:,slice_indx] = label_curr

    return img, label

class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res
        seed = 42
        self.rng = np.random.default_rng(seed)
        self.p = 0.5
        self.n = 2
        self.scale = (0.8, 1.2, 2)
        self.translate = (-0.2, 0.2, 2)
        self.shear = (-0.3, 0.3, 2)
        self.posterize = (4, 8.99, 2)
        self.contrast = (0.7, 1.3, 2)
        self.brightness = (0.7, 1.3, 2)
        self.sharpness = (0.1, 1.9, 2)

        self.create_ops()

    def create_ops(self):
        ops = [
            (shear_x, self.shear),
            (shear_y, self.shear),
            (scale, self.scale),
            (translate_x, self.translate),
            (translate_y, self.translate),
            (posterize, self.posterize),
            (contrast, self.contrast),
            (brightness, self.brightness),
            (sharpness, self.sharpness),
            (identity, (0, 1, 1)),
        ]

        self.ops = [op for op in ops if op[1][2] != 0]

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        if random.random() > 0.5:
            image, label = random_rotate(image, label)
        if random.random() > 0.5:
            image, label = adjust_light(image, label)
        if random.random() > 0.5:
            image, label = random_erasing(imgs=image, label=label, rng=self.rng)
        
        inds = self.rng.choice(len(self.ops), size=self.n, replace=False)
        for i in inds:
            op = self.ops[i]
            aug_func = op[0]
            aug_params = op[1]
            v = self.rng.uniform(aug_params[0], aug_params[1])

            image, label = aug_func(image, label, v)

        x, y, z = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1.0), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, 1.0), order=0)
        label_h, label_w, label_d = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w, 1.0), order=0)
        
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        image = image.permute(2, 0, 1)
        label = label.permute(2, 0, 1)
        low_res_label = low_res_label.permute(2, 0, 1)
        
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long()}
        return sample


class dataset_reader(Dataset):
    def __init__(self, base_dir, split, num_classes, transform=None):
        self.transform = transform 
        self.split = split
        
        self.data_dir = base_dir

        if split=="train":
            df = pd.read_csv(base_dir+'/training.csv')
            self.sample_list = [base_dir+'/'+sample_pth.split('/'+base_dir.split('/')[-1]+'/')[-1] for sample_pth in df["image_pth"]]
            self.masks_list = [base_dir+'/'+sample_pth.split('/'+base_dir.split('/')[-1]+'/')[-1] for sample_pth in df["mask_pth"]]
            self.num_classes = num_classes

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":

            data = read_image(self.sample_list[idx])
            data = np.clip(data, HU_min, HU_max)
            data = (data-HU_min)/(HU_max-HU_min)*255.0
            
            data = np.float32(data)
            data = (data - data_mean) / data_std
            data = (data-data.min())/(data.max()-data.min()+0.00000001)
            h, w, d = data.shape

            data = np.float32(data)
            
            mask = read_image(self.masks_list[idx])
            mask = np.float32(mask)
            
            if self.num_classes==12:
                mask[mask==13] = 12

            image = np.float32(data)
            label = np.float32(mask)

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
