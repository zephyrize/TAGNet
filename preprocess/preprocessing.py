from os.path import join
from scipy.ndimage import zoom
from skimage.transform import resize 
from utils.file_utils import check_dir, save_json, load_json, load_pickle, save_pickle
from preprocess.utils_func import *
from preprocess.cropping import Cropper

import os
import numpy as np
class Preprocessor(object):


    def __init__(self, root_dir, dataset='BileDuct', mode='train', process_type='2D'):
        super(Preprocessor, self).__init__()

        self.dataset = dataset
        self.mode = mode
        self.process_type = process_type

        self.dataset_root_dir = join(root_dir, dataset)
        self.crop_dir = join(self.dataset_root_dir, 'cropped_data')
        self.crop_dataset_json = join('datapath', 'crop_dataset.json')
        self.intensityproperties_file = join(self.crop_dir, 'intensityproperties.pkl')

        if not os.path.exists(self.crop_dataset_json):
            cropper_train = Cropper(root_dir, dataset, 'train')
            cropper_test = Cropper(root_dir, dataset, 'test')

            cropper_train.run()
            cropper_test.run()

        self.preprocess_dir = join(self.dataset_root_dir, 'preprocessed_data', process_type)
        self.preprocess_dataset_json = join('datapath', 'preprocess_dataset.json')
        
        check_dir(self.preprocess_dir)

        self.resize_ct = lambda data, x, y: zoom(data, (1.0, self.patch_size / x, self.patch_size / y), order=3)
        self.resize_seg = lambda data, x, y: zoom(data, (1.0, self.patch_size / x, self.patch_size / y), order=0)
        
        self.z_spacing = 1.0
        self.patch_size = 256


    def load_intensity(self):
        
        intensity = load_pickle(self.intensityproperties_file)

        mean_intensity = intensity['mean']
        std_intensity = intensity['sd']
        lower_bound = intensity['percentile_00_5']
        upper_bound = intensity['percentile_99_5']
        
        return mean_intensity, std_intensity, lower_bound, upper_bound

    def run(self, consecutive_slices=9):
        print('begin preprocessing...')

        crop_dataset_info = load_json(self.crop_dataset_json)

        if not os.path.exists(self.preprocess_dataset_json):
            preprocess_dataset_info = {}
            preprocess_dataset_info['2D'] = {}
            preprocess_dataset_info['2.5D'] = {}
            preprocess_dataset_info['3D'] = {}
            preprocess_dataset_info['name'] = crop_dataset_info['name']
        else:
            preprocess_dataset_info = load_json(self.preprocess_dataset_json)

        for idx, path_dic in enumerate(crop_dataset_info[self.mode]):

            case_name = path_dic['name']
            
            data = np.load(path_dic['crop_npy'])
            properties = load_pickle(path_dic['crop_pkl'])

            assert len(data.shape) == 4

            if self.process_type == '2D':
                data = self._resize(data)
                data = self.normalization(data)
                data, properties = trans25D(data, properties, consecutive_slices)
                
            elif self.process_type == '2.5D':
                data, properties = self.z_interpolation(data, properties)
                data, properties = self.get_new_crop(data, properties)
                data = self._resize(data)
                data = self.normalization(data)
                data, properties = trans25D(data, properties, consecutive_slices)

            elif self.process_type == '3D':
                data, properties = self.xyz_interpolation(data, properties)
                data, properties = self.get_new_crop(data, properties)
                data = self.normalization(data)

            # save preprocessed data
            np.save(join(self.preprocess_dir, "%s.npy" % case_name), data)
            # save properties pickle
            save_pickle(properties, join(self.preprocess_dir, "%s.pkl" % case_name))

            print('crop shape: ', properties['crop_shape'], ' new crop shape: ', data[0].shape)

            crop_dataset_info[self.mode][idx]['preprocess_npy'] = join(self.preprocess_dir, case_name + '.npy')
            crop_dataset_info[self.mode][idx]['preprocess_pkl'] = join(self.preprocess_dir, "%s.pkl" % case_name)

            print(case_name + ' preprocessed done...')

        preprocess_dataset_info[self.process_type][self.mode] = crop_dataset_info[self.mode]

        print('new data infomation have been stored successfully...')

        save_json(preprocess_dataset_info, file=self.preprocess_dataset_json)

        print('preprocess dataset json file have been saved done...\n')
        

    def z_interpolation(self, data, properties):

        origin_space = properties['spacing']
        new_spacing = np.array([self.z_spacing, origin_space[1], origin_space[2]])
        properties['new_spacing'] = new_spacing

        scale_rate = origin_space / new_spacing
        new_shape = np.round(data[0].shape * scale_rate).astype(np.int)

        new_data = np.array([
            resize(data[0].astype(np.float32), new_shape, order=3).astype(np.float32), # interpolate image
            resize(data[1].astype(np.float32), new_shape, order=0).astype(np.uint8), # interpolate mask
            resize(data[2].astype(np.float32), new_shape, order=0).astype(np.uint8) # interpolate label
        ])
        properties['interpolation_shape'] = new_data[0].shape
        
        return new_data, properties
    
    def xyz_interpolation(self, data, properties):
        pass
    
    def get_new_crop(self, data, properties):
        new_box = get_nonzero_box(data[1])
        new_box = extend_slices(new_box, 20, data.shape[1])
        new_data = crop(data, new_box)
        properties['new_box'] = new_box
        properties['new_crop_shape'] = new_data[0].shape

        return new_data, properties


    def _resize(self, data):
        
        new_data = np.array([
            self.resize_ct(data[i], data[i].shape[1], data[i].shape[2])
            for i in range(data.shape[0])
        ])

        return new_data

    def normalization(self, data):

        mean_intensity, std_intensity, lower_bound, upper_bound = self.load_intensity()
        data[0] = np.clip(data[0], lower_bound, upper_bound)
        data[0] = (data[0] - mean_intensity) / std_intensity
        return data


    def split_train_val(self):
        
        self.split_json_dir = join('datapath', 'split_train_val.json')
        if not os.path.exists(self.split_json_dir):
            split_json = {}
        else:
            split_json = load_json(self.split_json_dir)

        split_json[self.process_type] = {}

        preprocess_dataset_json = load_json(self.preprocess_dataset_json)

        split_json[self.process_type]['name'] = preprocess_dataset_json['name']
        if 'test' in preprocess_dataset_json[self.process_type].keys():
            split_json[self.process_type]['test'] = preprocess_dataset_json[self.process_type]['test']

        train_list, val_list = [], []

        for idx, dic in enumerate(preprocess_dataset_json[self.process_type]['train']):

            if dic['name'] == 'BileDuct_009' or dic['name'] == 'BileDuct_016' or dic['name'] == 'BileDuct_023':
                val_list.append(dic)
            else:
                train_list.append(dic)
        
        split_json[self.process_type]['train'] = train_list
        split_json[self.process_type]['val'] = val_list

        save_json(split_json, self.split_json_dir)

        print('dataset have been splited to train and val done...\n')
            



        


    