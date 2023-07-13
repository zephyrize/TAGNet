import sys
sys.path.append('../')

from os.path import join
from collections import OrderedDict
from utils.generate_data_index import generate_dataset_json
from utils.file_utils import save_json, load_json, save_pickle, check_dir
from preprocess.utils_func import *

import numpy as np
import SimpleITK as sitk


class Cropper:
    def __init__(self, root_dir, dataset='BileDuct', mode='train'):

        super(Cropper, self).__init__()
        self.dataset_root_dir = join(root_dir, dataset)
        self.raw_dir = join(self.dataset_root_dir, 'raw_data')

        self.dataset_json = join(self.raw_dir, 'dataset.json')

        self.crop_dir = join(self.dataset_root_dir, 'cropped_data')
        self.crop_dataset_json = join('datapath', 'crop_dataset.json')
        self.intensityproperties_file = join(self.crop_dir, 'intensityproperties.pkl')
        
        check_dir(self.crop_dir)

        self.dataset = dataset
        self.mode = mode

        if self.mode == 'train':
            generate_dataset_json(dataset=self.dataset)

        self.extend_slices = 20

    def run(self):
        dataset_info = load_json(self.dataset_json)
        '''
        if mode is train, then we get our dataset infor from dataset_json file and crop these data.
        if mode is test, which means train dataset have been cropped before and we cannot change original cropped json file. 
        What we do here is get test path information from dataset_json file and put it into cropped_data_json file.
        '''

        if self.mode == 'train':
            crop_dataset_info = dataset_info
        else:
            crop_dataset_info = load_json(self.crop_dataset_json)
            crop_dataset_info['test'] = dataset_info['test']

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
            
            # get liver box
            liver_box = get_nonzero_box(mask = mask_arr)
            # extend slices to 20 from z axis during training
            liver_box = extend_slices(liver_box, self.extend_slices, max_slices=ct_arr.shape[0])
            
            data = np.array([ct_arr, mask_arr, label_arr])

            # crop
            data = crop(data, liver_box)

            if self.mode == 'train':
                # forgroung sample
                temp_mask = data[-1] > 0
                voxels = list(data[0][temp_mask][::10])
                all_voxels += voxels
            else:
                voxels = None

            # save origin volume and cropping infor
            self.set_origin_properties(ct, liver_box=liver_box, origin_shape=ct_arr.shape, crop_shape=data[0].shape, voxels=voxels)
            
            # save cropped data and properties
            np.save(join(self.crop_dir, "%s.npy" % case_name), data)
            save_pickle(data = self.properties, path=join(self.crop_dir, "%s.pkl" % case_name))
            
            # add crop image path and crop label path into json file
            crop_dataset_info[self.mode][idx]['crop_npy'] = join(self.crop_dir, case_name + '.npy')
            crop_dataset_info[self.mode][idx]['crop_pkl'] = join(self.crop_dir, case_name + '.pkl')
            
            print('original shape: ', self.properties['size'], ' new shape: ', data[0].shape)
            print(case_name + ' cropped done...')

        if self.mode == 'train':
            
            median, mean, sd, mn, mx, percentile_99_5, percentile_00_5 = compute_stats(all_voxels)

            dataset_prop = OrderedDict()
            dataset_prop['median'] = median
            dataset_prop['mean'] = mean
            dataset_prop['sd'] = sd
            dataset_prop['mn'] = mn
            dataset_prop['mx'] = mx
            dataset_prop['percentile_99_5'] = percentile_99_5
            dataset_prop['percentile_00_5'] = percentile_00_5

            save_pickle(data=dataset_prop, path=self.intensityproperties_file)

            print('The dataset intensity propertties have been calculated and saved done...')

        save_json(crop_dataset_info, self.crop_dataset_json)

        print('Crop dataset json file have been saved done...\n')


    def set_origin_properties(self, ct, liver_box, origin_shape, crop_shape, voxels):

        self.properties = OrderedDict()
        self.properties['origin_box'] = liver_box
        
        self.properties['size'] = np.array(ct.GetSize())[[2, 1, 0]]
        # self.properties['origin'] = np.array(ct.GetOrigin())[[2, 1, 0]]
        self.properties['spacing'] = np.array(ct.GetSpacing())[[2, 1, 0]]

        self.properties['itk_origin'] = ct.GetOrigin()
        self.properties['itk_spacing'] = ct.GetSpacing()
        self.properties['itk_direction'] = ct.GetDirection()

        self.properties['origin_shape'] = origin_shape
        self.properties['crop_shape'] = crop_shape

        if voxels is not None:
            median, mean, sd, mn, mx, percentile_99_5, percentile_00_5 = compute_stats(voxels)
            self.properties['median'] = median
            self.properties['mean'] = mean
            self.properties['sd'] = sd
            self.properties['mn'] = mn
            self.properties['mx'] = mx
            self.properties['percentile_99_5'] = percentile_99_5
            self.properties['percentile_00_5'] = percentile_00_5

