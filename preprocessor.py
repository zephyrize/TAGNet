
'''
2D preprocess

'''

import os
import pickle
from matplotlib.pyplot import axis
import numpy as np
import SimpleITK as sitk

from collections import OrderedDict
from skimage.transform import resize 
from scipy.ndimage import zoom
from os.path import join
from utils.generate_data_index import generate_dataset_json
from utils.file_utils import save_json, load_json, load_pickle, save_pickle


class Preprocessor:

    def __init__(self, root_dir, dataset='BileDuct', mode='train'):
        '''
        root_dir: all datasets dir: '/home/sophia/zfx/data/' in ubuntu 20.04.

        dataset: dataset name.

        mode: 'train' or 'test. 
                if mode = 'train', then save all cropped and preprocessed data to check data conveniently.
                if mode = 'test', then do not save cropped data, just return a list of preprocessed data.
        '''
        self.dataset_root_dir = join(root_dir, dataset)
        self.raw_dir = join(self.dataset_root_dir, 'raw_data')
        
        self.crop_dir = join(self.dataset_root_dir, 'cropped_data')
        self.crop_dataset_json = join(self.crop_dir, 'crop_dataset.json')
        self.preprocess_dir = join(self.dataset_root_dir, 'preprocessed_data')
        self.preprocess_dataset_json = join(self.preprocess_dir, 'preprocess_dataset.json')

        self.dataset = dataset
        self.mode = mode

        self.dataset_json = join(self.raw_dir, 'dataset.json')
        
        # check if dataset json file exist, if not, generate it
        # if not os.path.exists(self.dataset_json):
        generate_dataset_json(dataset=self.dataset)
        
        self.intensityproperties_file = join(self.crop_dir, 'intensityproperties.pkl')
        
        self.check_file()

        self.patch_size = 256
        self.con_slices = 9
        self.properties = None

        self.resize_ct = lambda data, x, y: zoom(data, (1.0, self.patch_size / x, self.patch_size / y), order=3)
        self.resize_seg = lambda data, x, y: zoom(data, (1.0, self.patch_size / x, self.patch_size / y), order=0)

    def check_file(self):
        
        if not os.path.exists(self.crop_dir):
            os.mkdir(self.crop_dir)
        if not os.path.exists(self.preprocess_dir):
            os.mkdir(self.preprocess_dir)

    def get_nonzero_box(self, mask):

        mask_voxel_coords = np.where(mask != 0)
        minzidx = int(np.min(mask_voxel_coords[0]))
        maxzidx = int(np.max(mask_voxel_coords[0])) + 1
        minxidx = int(np.min(mask_voxel_coords[1])) 
        maxxidx = int(np.max(mask_voxel_coords[1])) + 1
        minyidx = int(np.min(mask_voxel_coords[2]))
        maxyidx = int(np.max(mask_voxel_coords[2])) + 1

        return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


    def save_nii_file(self, ct_arr, label_arr, label_file):
        
        new_ct = sitk.GetImageFromArray(ct_arr)
        new_ct.SetDirection(self.properties['itk_direction'])
        new_ct.SetOrigin(self.properties['itk_origin'])
        new_ct.SetSpacing(tuple(self.median_spacing[[2,1,0]]))

        new_label = sitk.GetImageFromArray(label_arr)
        new_label.SetDirection(self.properties['itk_direction'])
        new_label.SetOrigin(self.properties['itk_origin'])
        new_label.SetSpacing(tuple(self.median_spacing[[2,1,0]]))

        sitk.WriteImage(new_ct, os.path.join(self.save_dir, 'ct', label_file))
        sitk.WriteImage(new_label, os.path.join(self.save_dir, 'label', label_file))

        print('save ' + label_file.split('.')[0] + ' nii file done...')


    def _compute_stats(self, voxels):
        
        if len(voxels) == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        median = np.median(voxels)
        mean = np.mean(voxels)
        sd = np.std(voxels)
        mn = np.min(voxels)
        mx = np.max(voxels)
        percentile_99_5 = np.percentile(voxels, 99.5)
        percentile_00_5 = np.percentile(voxels, 00.5)

        return median, mean, sd, mn, mx, percentile_99_5, percentile_00_5
    

    def set_origin_properties(self, ct, liver_box, origin_shape, crop_shape, voxels):

        self.properties = OrderedDict()
        self.properties['liver_box'] = liver_box
        
        self.properties['size'] = np.array(ct.GetSize())[[2, 1, 0]]
        # self.properties['origin'] = np.array(ct.GetOrigin())[[2, 1, 0]]
        # self.properties['spacing'] = np.array(ct.GetSpacing())[[2, 1, 0]]

        self.properties['itk_origin'] = ct.GetOrigin()
        self.properties['itk_spacing'] = ct.GetSpacing()
        self.properties['itk_direction'] = ct.GetDirection()
        self.properties['origin_shape'] = origin_shape
        self.properties['crop_shape'] = crop_shape

        if voxels is not None:
            median, mean, sd, mn, mx, percentile_99_5, percentile_00_5 = self._compute_stats(voxels)
            self.properties['median'] = median
            self.properties['mean'] = mean
            self.properties['sd'] = sd
            self.properties['mn'] = mn
            self.properties['mx'] = mx
            self.properties['percentile_99_5'] = percentile_99_5
            self.properties['percentile_00_5'] = percentile_00_5

    
    def crop_data(self):

        dataset_info = load_json(self.dataset_json)

        crop_dataset_info = dataset_info

        print('begin data cropping......')

        all_voxels = []

        for idx, path_dic in enumerate(dataset_info[self.mode]):

            case_name = path_dic['name']

            ct = sitk.ReadImage(path_dic['image'])
            mask = sitk.ReadImage(path_dic['mask'])
            label = sitk.ReadImage(path_dic['label'])

            ct_arr = sitk.GetArrayFromImage(ct)
            mask_arr = sitk.GetArrayFromImage(mask)
            label_arr = sitk.GetArrayFromImage(label)

            label_arr[label_arr > 1] = 0
            mask_arr[mask_arr > 0] = 1

            assert len(np.unique(label_arr)) == 2

            liver_box = self.get_nonzero_box(mask = mask_arr)

            if self.mode == 'train' and self.dataset == 'BileDuct':
                if liver_box[0][0] -20 >= 0:
                    liver_box[0][0] -= 20
                else:
                    liver_box[0][0] = 0
            
            # get liver box
            ct_liver = ct_arr[liver_box[0][0]:liver_box[0][1], liver_box[1][0]:liver_box[1][1], liver_box[2][0]:liver_box[2][1]].copy()
            label_liver = label_arr[liver_box[0][0]:liver_box[0][1], liver_box[1][0]:liver_box[1][1], liver_box[2][0]:liver_box[2][1]].copy()

            if self.mode == 'train':
                # forgroung sample
                temp_mask = label_liver > 0
                voxels = list(ct_liver[temp_mask][::10])
                all_voxels += voxels
            else:
                voxels = None

            self.set_origin_properties(ct, liver_box=liver_box, origin_shape=ct_arr.shape, crop_shape=ct_liver.shape, voxels=voxels)

            ct_liver, label_liver = np.expand_dims(ct_liver, axis=0), np.expand_dims(label_liver, axis=0)
            all_data = np.vstack([ct_liver, label_liver])

            # save cropped data and properties
            np.save(os.path.join(self.crop_dir, "%s.npy" % case_name), all_data)
            save_pickle(data = self.properties, path=join(self.crop_dir, "%s.pkl" % case_name))
            
            # add crop image path and crop label path into json file
            crop_dataset_info[self.mode][idx]['crop_npy'] = join(self.crop_dir, case_name + '.npy')
            crop_dataset_info[self.mode][idx]['crop_pkl'] = join(self.crop_dir, case_name + '.pkl')
            
            print('original shape: ', self.properties['size'], ' new shape: ', ct_liver[0].shape)
            print(case_name + ' cropped done...')
        
        if self.mode == 'train':
            median, mean, sd, mn, mx, percentile_99_5, percentile_00_5 = self._compute_stats(all_voxels)

            dataset_prop = OrderedDict()
            dataset_prop['median'] = median
            dataset_prop['mean'] = mean
            dataset_prop['sd'] = sd
            dataset_prop['mn'] = mn
            dataset_prop['mx'] = mx
            dataset_prop['percentile_99_5'] = percentile_99_5
            dataset_prop['percentile_00_5'] = percentile_00_5

            save_pickle(data=dataset_prop, path=join(self.intensityproperties_file))

            print('The dataset intensity propertties have been calculated and saved done...')

        save_json(crop_dataset_info, join(self.crop_dir, 'crop_dataset.json'))

        print('Crop dataset json file have been saved done...\n')

    def preprocess_data(self):
        
        print('begin preprocessing...')

        dataset_prop = pickle.load(open(self.intensityproperties_file, 'rb'))

        mean_intensity = dataset_prop['mean']
        std_intensity = dataset_prop['sd']
        lower_bound = dataset_prop['percentile_00_5']
        upper_bound = dataset_prop['percentile_99_5']

        crop_dataset_info = load_json(self.crop_dataset_json)

        if self.mode == 'test':
            preprocess_dataset_info = load_json(self.preprocess_dataset_json)
            preprocess_dataset_info['test'] = crop_dataset_info['test']
        else:
            preprocess_dataset_info = crop_dataset_info

        for idx, path_dic in enumerate(crop_dataset_info[self.mode]):

            case_name = path_dic['name']

            data = np.load(path_dic['crop_npy'])

            assert len(data.shape) == 4

            # resize to 256 256
            
            new_ct = self.resize_ct(data[0], data[0].shape[1], data[0].shape[2])
            new_label = self.resize_seg(data[-1], data[-1].shape[1], data[-1].shape[2])

            # normalization 
            
            new_ct = np.clip(new_ct, lower_bound, upper_bound)
            new_ct = (new_ct - mean_intensity) / std_intensity
            
            # trans to 2.5D: new_ct shape: (slices, self.con_slices, x, y); new_label shape: (slices, 1, x, y)
            new_ct, new_label = self.trans25D(new_ct, new_label)

            new_ct, new_label = np.expand_dims(new_ct, axis=0), np.expand_dims(new_label, axis=0)
            new_data = np.vstack([new_ct, new_label])
            
            # save data and properties
            np.save(os.path.join(self.preprocess_dir, "%s.npy" % case_name), new_data)

            preprocess_dataset_info[self.mode][idx]['preprocess_npy'] = join(self.preprocess_dir, case_name + '.npy')
            preprocess_dataset_info[self.mode][idx]['preprocess_pkl'] = path_dic['crop_pkl']

            print(case_name + ' preprocessed done...')

        print('new data infomation have been stored successfully...')

        save_json(preprocess_dataset_info, join(self.preprocess_dir, 'preprocess_dataset.json'))
        print('preprocess dataset json file have been saved done...')
        

    def trans25D(self, ct, label):
        
        assert self.con_slices % 2 == 1

        slices_num = ct.shape[0]
        #zeros_pad = np.zeros([self.patch_size, self.patch_size]).astype(ct.dtype)
        left_pad, right_pad = ct[0], ct[slices_num-1]

        expand_slices = self.con_slices // 2
        expand_ct = np.insert(ct, [0]*expand_slices, left_pad, axis=0)
        expand_ct = np.insert(expand_ct, [expand_ct.shape[0]]*expand_slices, right_pad, axis=0)
        # expand_ct = np.insert(ct, [0]*expand_slices + [slices_num]*expand_slices, zeros_pad, axis = 0)

        new_ct = np.array([expand_ct[i:i+self.con_slices].copy() for i in range(slices_num)])

        return new_ct, np.repeat(np.expand_dims(label, 1), self.con_slices, 1)

    def generate_path_index(self):

        data_name_list = [f for f in os.listdir(self.preprocess_dir) if f.find('npy') != -1]
        data_num = len(data_name_list)

        print('data_num: ', data_num)

        self.write_name_list(data_name_list, "data_path_info.txt")

        print('path index have been created successfully')
    
    def write_name_list(self, name_list, file_name):
        f = open(os.path.join(self.preprocess_dir, file_name), 'w')

        for name in name_list:
            data_path = os.path.join(self.preprocess_dir, name.replace('pkl', 'npy'))
            properties_path = os.path.join(self.preprocess_dir, name)
            f.write(data_path + ' ' + properties_path + "\n")

        f.close()
    
    def split_train_val(self):
        
        self.split_json_dir = join(self.preprocess_dir, 'split_train_val.json')

        preprocess_dataset_json = load_json(self.preprocess_dataset_json)

        split_json = {}

        split_json['name'] = preprocess_dataset_json['name']
        split_json['test'] = preprocess_dataset_json['test']

        train_list, val_list = [], []
        for idx, dic in enumerate(preprocess_dataset_json['train']):

            if dic['name'] == 'BileDuct_009' or dic['name'] == 'BileDuct_016' or dic['name'] == 'BileDuct_023':
                val_list.append(dic)
            else:
                train_list.append(dic)
        
        split_json['train'] = train_list
        split_json['val'] = val_list

        save_json(split_json, self.split_json_dir)

        print('dataset have been splited to train and val done...')

if __name__ == '__main__':
    
    from config import args

    root_dir = args.data_root_dir
    dataset = args.dataset

    preprocess = Preprocessor(root_dir, dataset=dataset, mode='train')
    
    preprocess.crop_data()
    preprocess.preprocess_data()

    preprocess.split_train_val()
