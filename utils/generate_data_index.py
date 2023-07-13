import json
from os.path import join
import numpy as np
import os
from utils.file_utils import save_json, subfiles
from config import args

def get_identifiers_from_splitted_files(folder: str):

    uniques = np.unique([i.split('.')[0] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques

def generate_dataset_json(dataset):

    root_dir = args.data_root_dir

    raw_dir = join(root_dir, dataset, 'raw_data')

    imagesTr_dir = join(raw_dir, 'imagesTr')
    labelsTr_dir = join(raw_dir, 'labelsTr')
    imagesTs_dir = join(raw_dir, 'imagesTs')
    labelsTs_dir = join(raw_dir, 'labelsTs')

    mask_dir = join(raw_dir, 'mask')

    
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)
    test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)

    json_dict = {}
    json_dict['name'] = dataset
    json_dict['train'] = [{ 'name' : name.split('_volume')[0],
                            'image': join(imagesTr_dir, name + '.nii.gz'), 
                            'mask': join(mask_dir, name.replace('volume', 'liver.nii.gz')),
                            'label': join(labelsTr_dir, name.replace('volume', 'label.nii.gz'))} for name 
                            in 
                            train_identifiers]


    json_dict['test'] = [{  'name' : name.split('_volume')[0],
                            'image': join(imagesTs_dir, name + '.nii.gz'), 
                            'mask': join(mask_dir, name.replace('volume', 'liver.nii.gz')),
                            'label': join(labelsTs_dir, name.replace('volume', 'label.nii.gz'))} for name 
                            in 
                            test_identifiers]

    save_json(json_dict, join(raw_dir, 'dataset.json'))

    print('raw dataset json file have been generated done...')

if __name__ == '__main__':

    generate_dataset_json('BileDuct')

