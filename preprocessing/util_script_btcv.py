import shutil
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
from torch.nn import functional as F
import pickle
import random
import pandas as pd
from tqdm import tqdm
import cv2
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom

base_dir = '/mnt/weka/wekafs/rad-megtron/cchen'


def get_all_5slice():

    save_pth = base_dir + "/synapseCT/Training/2D_all_5slice"
    data_pth = base_dir + "/synapseCT/Training/img"

    data_fd_list = os.listdir(data_pth)
    data_fd_list = [data_fd for data_fd in data_fd_list if data_fd.startswith("img") and data_fd.endswith(".nii.gz")]
    data_fd_list.sort()

    cnt = 0
    for data_fd_indx, data_fd in enumerate(data_fd_list):
        case_id = data_fd[3:7]

        if not os.path.exists(save_pth+'/'+case_id):
            os.makedirs(save_pth+'/'+case_id)
            os.mkdir(save_pth+'/'+case_id+'/images')
            os.mkdir(save_pth+'/'+case_id+'/masks')
    
        img_obj = nib.load(data_pth + '/' + data_fd)
        img_arr = img_obj.get_fdata()

        #load mask
        mask_obj = nib.load(data_pth.replace("/img", "/label") + '/' + data_fd.replace("img", "label"))
        mask_arr = mask_obj.get_fdata()

        img_arr = np.concatenate((img_arr[:, :, 0:1], img_arr[:, :, 0:1], img_arr, img_arr[:, :, -1:], img_arr[:, :, -1:]), axis=-1)
        mask_arr = np.concatenate((mask_arr[:, :, 0:1], mask_arr[:, :, 0:1], mask_arr, mask_arr[:, :, -1:], mask_arr[:, :, -1:]), axis=-1)

        for slice_indx in range(2, img_arr.shape[2]-2):
            
            slice_arr = img_arr[:,:,slice_indx-2: slice_indx+3]
            slice_arr = np.flip(np.rot90(slice_arr, k=1, axes=(0, 1)), axis=1)

            mask_arr_2D = mask_arr[:,:,slice_indx-2: slice_indx+3]
            mask_arr_2D = np.flip(np.rot90(mask_arr_2D, k=1, axes=(0, 1)), axis=1)

            with open(save_pth+'/'+case_id+'/images'+'/2Dimage_'+'{:04d}'.format(slice_indx-2)+'.pkl', 'wb') as file:
                pickle.dump(slice_arr, file)

            with open(save_pth+'/'+case_id+'/masks'+'/2Dmask_'+'{:04d}'.format(slice_indx-2)+'.pkl', 'wb') as file:
                pickle.dump(mask_arr_2D, file)

        cnt += 1

def get_csv():
    save_pth = base_dir + "/synapseCT/Training/2D_all_5slice"
    
    training_csv = save_pth+'/training.csv'
    test_csv = save_pth+'/test.csv'

    data_fd_list = os.listdir(save_pth)
    data_fd_list = [data_fd for data_fd in data_fd_list if data_fd.startswith('00') and '.' not in data_fd]
    
    random.shuffle(data_fd_list)
    random.shuffle(data_fd_list)
    random.shuffle(data_fd_list)
    random.shuffle(data_fd_list)
    random.shuffle(data_fd_list)

    test_fd_list = ['0035', '0036', '0037', '0038', '0039', '0040']

    training_fd_list = list(set(data_fd_list)-set(test_fd_list))

    path_list_all = []
    for data_fd in training_fd_list:
        slice_list = os.listdir(save_pth+'/'+data_fd+'/images')
        slice_pth_list = [data_fd+'/images/'+slice for slice in slice_list]
        path_list_all = path_list_all + slice_pth_list

    random.shuffle(path_list_all)
    random.shuffle(path_list_all)
    random.shuffle(path_list_all)
    random.shuffle(path_list_all)
    random.shuffle(path_list_all)
    df = pd.DataFrame(path_list_all, columns=['image_pth'])
    df['mask_pth'] = path_list_all
    df['mask_pth'] = df['mask_pth'].apply(lambda x: x.replace('/images/2Dimage_', '/masks/2Dmask_'))

    df.to_csv(training_csv, index=False)

    path_list_all = []
    for data_fd in test_fd_list:
        slice_list = os.listdir(save_pth+'/'+data_fd+'/images')
        slice_list.sort()
        slice_pth_list = [data_fd+'/images/'+slice for slice in slice_list]
        path_list_all = path_list_all + slice_pth_list

    df = pd.DataFrame(path_list_all, columns=['image_pth'])
    df['mask_pth'] = path_list_all
    df['mask_pth'] = df['mask_pth'].apply(lambda x: x.replace('/images/2Dimage_', '/masks/2Dmask_'))

    df.to_csv(test_csv, index=False)


def get_data_statistics():
    HU_min = -200
    HU_max = 250

    data_pth = base_dir + "/synapseCT/Training/img"
    data_fd_list = os.listdir(data_pth)
    data_fd_list = [data_fd for data_fd in data_fd_list if data_fd.startswith("img")]
    data_fd_list.sort()

    mean_val_list = []
    std_val_list = []
    psum_list = []
    psum_sq_list = []
    count_list = []
    for data_fd in tqdm(data_fd_list):
        
        img_obj = nib.load(data_pth + '/' + data_fd)
        img_arr = img_obj.get_fdata()

        # preprocessing
        img_arr[img_arr<=HU_min] = HU_min
        img_arr[img_arr>=HU_max] = HU_max
        img_arr = (img_arr-HU_min)/(HU_max-HU_min)*255.0

        psum = np.sum(img_arr)
        psum_sq = np.sum(img_arr ** 2)

        mean_val = np.mean(img_arr)
        std_val = np.std(img_arr)

        mean_val_list.append(mean_val)
        std_val_list.append(std_val)

        psum_list.append(psum)
        psum_sq_list.append(psum_sq)

        count_list.append(img_arr.shape[0]*img_arr.shape[1]*img_arr.shape[2])

    psum_tot = 0.0
    psum_sq_tot = 0.0
    count_tot = 0.0
    for i in range(len(psum_list)):
        psum_tot += psum_list[i]
        psum_sq_tot += psum_sq_list[i]
        count_tot += count_list[i]

    total_mean = psum_tot / count_tot # 50.21997497685108
    total_var  = (psum_sq_tot / count_tot) - (total_mean ** 2)
    total_std  = np.sqrt(total_var) # 68.47153712416372

    print(total_mean, total_std)


if __name__=="__main__":
    get_all_5slice()
    get_csv()
    get_data_statistics()
