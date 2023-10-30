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
import matplotlib.pyplot as plt

HU_min, HU_max = -50, 200
data_mean = 30.52402855024227
data_std = 64.07289127027198

def transform_bounding_box(bbox, shape, k, axis):
    """
    Transforms the bounding box coordinates based on the rotation and flip operations.
    bbox: tuple (x_min, y_min, x_max, y_max)
    shape: tuple (height, width) of the image/label
    k: number of 90-degree rotations
    axis: axis along which to flip (0 for vertical, 1 for horizontal)
    """
    x_min, y_min, x_max, y_max = bbox
    height, width = shape

    # Apply rotation
    for _ in range(k):
        x_min, y_min, x_max, y_max = y_min, width - x_max, y_max, width - x_min
        height, width = width, height  # Swap height and width for the next iteration

    # Apply flip
    if axis == 0:
        y_min, y_max = height - y_max, height - y_min
    elif axis == 1:
        x_min, x_max = width - x_max, width - x_min

    return x_min, y_min, x_max, y_max

def rotate_point(x, y, angle, cx, cy):
    """
    Rotates a point (x, y) about a center (cx, cy) by angle degrees.
    """
    angle = np.radians(angle)
    x_new = cx + (x - cx) * np.cos(angle) - (y - cy) * np.sin(angle)
    y_new = cy + (x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)
    return x_new, y_new

def rotate_bounding_box(bbox, angle, shape):
    """
    Rotates the bounding box by angle degrees.
    bbox: tuple (x_min, y_min, x_max, y_max)
    angle: rotation angle in degrees
    shape: tuple (height, width) of the image/label
    """
    x_min, y_min, x_max, y_max = bbox
    height, width = shape
    cx, cy = width // 2, height // 2

    # Rotate the four corners of the bounding box
    corners = [(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]
    rotated_corners = [rotate_point(x, y, angle, cx, cy) for x, y in corners]

    # Find the new bounding box
    x_coords, y_coords = zip(*rotated_corners)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    return x_min, y_min, x_max, y_max

def read_image(path):
    with open(path, 'rb') as file:
        img = pickle.load(file)
        return img

def random_rot_flip(image, label, bbox):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(0, 1))
    label = np.rot90(label, k, axes=(0, 1))
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()

    # Transform the bounding box
    new_bbox = transform_bounding_box(bbox[...,0], image.shape[:2], k, axis)
    new_bbox = np.tile(new_bbox, (label.shape[-1], 1))
    new_bbox = np.moveaxis(new_bbox, 0, -1)

    return image, label, new_bbox

def random_rotate(image, label, bbox):
    angle = np.random.randint(-15, 15)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    
    # Rotate the bounding box
    new_bbox = rotate_bounding_box((bbox[...,0][1], bbox[...,0][0], bbox[...,0][3], bbox[...,0][2]), angle, image.shape[:2])
    new_bbox = (new_bbox[1], new_bbox[0], new_bbox[3], new_bbox[2])
    new_bbox = np.tile(new_bbox, (label.shape[-1], 1))
    new_bbox = np.moveaxis(new_bbox, 0, -1)

    return image, label, new_bbox

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

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 

class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res
        seed = 42
        self.rng = np.random.default_rng(seed)
        self.p = 0.5
        self.n = 2
        self.scale = (0.8, 1.2, 0)
        self.translate = (-0.2, 0.2, 0)
        self.shear = (-0.3, 0.3, 0)
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
        image, label, bbox = sample['image'], sample['label'], sample['prompt']


        if random.random() > 0.5:
            image, label, bbox = random_rot_flip(image, label, bbox)
        if random.random() > 0.5:
            image, label, bbox = random_rotate(image, label, bbox)
        if random.random() > 0.5:
            image, label = adjust_light(image, label)
        
        inds = self.rng.choice(len(self.ops), size=self.n, replace=False)
        for i in inds:
            op = self.ops[i]
            aug_func = op[0]
            aug_params = op[1]
            v = self.rng.uniform(aug_params[0], aug_params[1])

            image, label = aug_func(image, label, v)

        x, y, z = image.shape
        label_h, label_w, label_d = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w, 1.0), order=0)
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        image = image.permute(2, 0, 1)
        label = label.permute(2, 0, 1)
        low_res_label = low_res_label.permute(2, 0, 1)
        bbox = torch.from_numpy(np.array(bbox).astype(np.float32))
        bbox = bbox.permute(-1, 0)
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long(), 'prompt': bbox}
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
            
            prompt_pth = self.masks_list[idx].replace('/masks/2Dmask_', '/boxprompts/2Dprompt_')
            boxprompt = read_image(prompt_pth)

            image = np.float32(data)
            label = np.float32(mask)
            boxprompt = np.float32(boxprompt)



        sample = {'image': image, 'label': label, 'prompt': boxprompt}
        if self.transform:
            sample = self.transform(sample)
            
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
