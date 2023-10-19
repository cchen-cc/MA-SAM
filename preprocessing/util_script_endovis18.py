import shutil
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
import pickle
import random
import pandas as pd
from tqdm import tqdm
# import cv2
import SimpleITK as sitk
from scipy.ndimage.interpolation import zoom
import json
from PIL import Image

base_dir = "/mnt/weka/wekafs/rad-megtron/cchen"

# Define RGB values and their corresponding mask indices
color_mappings = {
    (0, 255, 0): 1,
    (0, 255, 255): 2,
    (125, 255, 12): 3,
    (255, 55, 0): 4,
    (24, 55, 125): 5,
    (187, 155, 25): 6,
    (0, 255, 125): 7,
    (255, 255, 125): 8,
    (123, 15, 175): 9,
    (124, 155, 5): 10,
    (12, 255, 141): 11
}

def organize_data():
    save_pth = base_dir + '/' + 'Dataset907_endovis18'

    os.makedirs(save_pth+'/imagesTr', exist_ok=True)
    os.makedirs(save_pth+'/labelsTr', exist_ok=True)

    data_pth_all = [base_dir + '/' + 'endovis18/Train',
                    base_dir + '/' + 'endovis18/Test',
                    ]

    for data_pth in data_pth_all:
        data_fd_list = os.listdir(data_pth)
        data_fd_list = [data_fd for data_fd in data_fd_list if data_fd.startswith('seq')]
        data_fd_list.sort()

        for data_fd in data_fd_list:
            print(data_fd)
            if data_pth.split('/')[-1]=='Train':
                patient_ID = '00' + "{:02d}".format(int(data_fd.split('_')[-1]))
            elif data_pth.split('/')[-1]=='Test':
                patient_ID = '00' + "{:02d}".format(int(data_fd.split('_')[-1])+20)

            filename_list = os.listdir(data_pth+'/'+data_fd+'/labels')
            filename_list = [filename for filename in filename_list if filename.endswith('.png')]
            filename_list.sort()

            for filename in filename_list:

                file_ID = filename.split('.png')[0][5:]

                label_obj = Image.open(data_pth + '/' + data_fd + '/labels' + '/' + filename)
                label_arr = np.array(label_obj)

                mask = np.zeros_like(label_arr[:,:,0])  

                for color, index in color_mappings.items():
                    condition = (label_arr[:,:,0] == color[0]) & (label_arr[:,:,1] == color[1]) & (label_arr[:,:,2] == color[2])
                    mask[condition] = index

                mask_obj = Image.fromarray(mask.astype(np.uint8))
                mask_obj.save(save_pth+'/labelsTr/endovis_'+patient_ID+file_ID+'.png')

                shutil.copy(data_pth + '/' + data_fd + '/left_frames' + '/' + filename, save_pth+'/imagesTr/endovis_'+patient_ID+file_ID+'_0000.png')


def get_all_5slice():

    save_pth = base_dir + '/endovis18/2D_all_5slice'
    data_pth_all = [base_dir + '/Dataset907_endovis18',
                    ]

    for data_pth in data_pth_all:
        data_fd_list = os.listdir(data_pth+'/imagesTr')
        data_fd_list = [data_fd.split('_')[1][0:4] for data_fd in data_fd_list if data_fd.endswith('.png')]
        data_fd_list = list(set(data_fd_list))
        data_fd_list.sort()

        cnt = 0
        for data_fd_indx, data_fd in enumerate(data_fd_list):
            case_id = data_fd

            if not os.path.exists(save_pth+'/'+case_id):
                os.makedirs(save_pth+'/'+case_id)
                os.mkdir(save_pth+'/'+case_id+'/images')
                os.mkdir(save_pth+'/'+case_id+'/masks')

            filename_all = os.listdir(data_pth+'/imagesTr')
            filename_all = [filename for filename in filename_all if filename.endswith('.png') and filename.split('_')[1][0:4]==case_id]
            filename_all.sort()

            print(case_id)

            img_arr, mask_arr = [], []
            for filename in filename_all:
                
                image_slice = np.array(Image.open(data_pth+'/imagesTr/'+filename))
                mask_slice = np.array(Image.open(data_pth+'/labelsTr/'+filename.replace('_0000.png', '.png')))
                h, w  = image_slice.shape[0], image_slice.shape[1]
                out_h, out_w = 512, 512
                if h != 512 or w !=512:
                    image_slice = zoom(image_slice, (out_h / h, out_w / w, 1.0), order=3)
                    mask_slice = zoom(mask_slice, (out_h / h, out_w / w), order=0)

                img_arr.append(image_slice)
                mask_arr.append(mask_slice)

            img_arr = np.transpose(np.array(img_arr), (1, 2, 3, 0))
            mask_arr = np.transpose(np.array(mask_arr), (1, 2, 0))
        
            print(case_id)
            
            img_arr = np.concatenate((img_arr[:, :, :, 0:1], img_arr[:, :, :, 0:1], img_arr, img_arr[:, :, :, -1:], img_arr[:, :, :, -1:]), axis=-1)
            mask_arr = np.concatenate((mask_arr[:, :, 0:1], mask_arr[:, :, 0:1], mask_arr, mask_arr[:, :, -1:], mask_arr[:, :, -1:]), axis=-1)    

            for slice_indx in range(2, img_arr.shape[-1]-2):
                
                slice_arr = img_arr[:,:,:,slice_indx-2: slice_indx+3]
                slice_arr = np.flip(np.rot90(slice_arr, k=1, axes=(0, 1)), axis=1)

                mask_arr_2D = mask_arr[:,:,slice_indx-2: slice_indx+3]
                mask_arr_2D = np.flip(np.rot90(mask_arr_2D, k=1, axes=(0, 1)), axis=1)

                with open(save_pth+'/'+case_id+'/images'+'/2Dimage_'+'{:04d}'.format(slice_indx-2)+'.pkl', 'wb') as file:
                    pickle.dump(slice_arr, file)

                with open(save_pth+'/'+case_id+'/masks'+'/2Dmask_'+'{:04d}'.format(slice_indx-2)+'.pkl', 'wb') as file:
                    pickle.dump(mask_arr_2D, file)

            cnt += 1


def get_csv():
            
    save_pth = base_dir + '/endovis18/2D_all_5slice'
    
    training_csv = save_pth+'/training.csv'
    validation_csv = save_pth+'/validation.csv'
    test_csv = save_pth+'/test.csv'

    data_fd_list = os.listdir(save_pth)
    data_fd_list = [data_fd for data_fd in data_fd_list if data_fd.startswith('00') and '.' not in data_fd]
    
    random.shuffle(data_fd_list)
    random.shuffle(data_fd_list)
    random.shuffle(data_fd_list)
    random.shuffle(data_fd_list)
    random.shuffle(data_fd_list)

    test_fd_list = ['0021', '0022', '0023', '0024']

    training_fd_list = list(set(data_fd_list)-set(test_fd_list))
    validation_fd_list = random.sample(test_fd_list, 4)

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
    for data_fd in validation_fd_list:
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

    df.to_csv(validation_csv, index=False)

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
    debugc = 1


if __name__=="__main__":
    organize_data()
    get_all_5slice()
    get_csv()
