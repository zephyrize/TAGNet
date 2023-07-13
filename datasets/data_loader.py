import os
import sys
sys.path.append('../')

from os.path import join

import cv2
import torch 
import numpy as np

from torchvision import transforms
from datasets.utils_loader import *
from utils.file_utils import load_json
from torch.utils.data import Dataset, DataLoader

from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


class data_loader(Dataset):
    def __init__(self, args, filename, mode='train', process_type='2D', sample_slices=3):

        self.args = args
        self.mode = mode
        self.type = process_type

        self.dataset_dir = join(args.data_root_dir, args.dataset)
        if self.mode == 'train':
            self.filename_list = load_json(join(self.dataset_dir, 'preprocessed_data', filename))['train']
        elif self.mode == 'val':
            self.filename_list = load_json(join(self.dataset_dir, 'preprocessed_data', filename))['val']
        elif self.mode == 'test':
            self.filename_list = load_json(join(self.dataset_dir, 'preprocessed_data', filename))['test']

        self.image, self.label = self._get_image_list(self.filename_list)

        self.extend_slice = sample_slices // 2

        self.data_aug = iaa.Sequential([
                        iaa.Affine(
                            scale=(0.5, 1.2),
                            rotate=(-15, 15)
                        ),  # rotate the image
                        iaa.Flipud(0.5),
                        iaa.PiecewiseAffine(scale=(0.01, 0.05)),
                        iaa.Sometimes(
                            0.1,
                            iaa.GaussianBlur((0.1, 1.5)),
                        ),
                        iaa.Sometimes(
                            0.1,
                            iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                        )
                    ])
        self.transforms = transforms.Compose([ToTensor()])

    def __getitem__(self, index):
    
        if self.mode == 'train':
            
            mid_slice = self.image.shape[1] // 2

            image = self.image[index][mid_slice-self.extend_slice:mid_slice+self.extend_slice+1, ...].copy()
            label = self.label[index][mid_slice,...].copy()
            
            image = image.transpose(1, 2, 0)
            segmap = SegmentationMapsOnImage(np.uint8(label), shape=(256, 256))

            # data augmentation
            image, label = self.data_aug(image=image, segmentation_maps=segmap)

            image, label = image.copy(), label.copy()

            image = image.transpose(2, 0, 1)
            label = label.get_arr()
            edge = self.get_edge(label)
        
        else:
            mid_slice = self.image[index].shape[1] // 2
            image = self.image[index][:, mid_slice-self.extend_slice:mid_slice+self.extend_slice+1, ...].copy()
            label = self.label[index][:, mid_slice, ...].copy()

            edge = self.get_edge(label)
            image, label, edge = torch.from_numpy(image), torch.from_numpy(label).unsqueeze(1), torch.from_numpy(edge).unsqueeze(1)

        sample = {'image': image, 'label': label, 'edge': edge}
        
        if self.mode == 'train':
            sample = self.transforms(sample)
        
        return sample
        
    def __len__(self):
        return self.image.shape[0]

    def get_edge(self, label):
        
        if len(label.shape) == 2:
            edge = cv2.Canny(np.uint8(label), 0, 1)
            edge[edge > 0] = 1
        elif len(label.shape) == 3:
            edge = np.empty_like(label)
            for idx in range(label.shape[0]):
                temp_edge = cv2.Canny(np.uint8(label[idx]), 0, 1)
                temp_edge[temp_edge > 0] = 1
                edge[idx] = temp_edge

        return edge
    
    def _get_image_list(self, filename_list):
        
        ct_list, label_list = [], []

        for dic in filename_list:

            data = np.load(dic['preprocess_npy'])

            if self.mode == 'train':
                ct_list.extend(data[0])
                label_list.extend(data[-1])
            else:
                ct_list.append(data[0])
                label_list.append(data[-1])
        return np.array(ct_list), np.array(label_list)

if __name__ == '__main__':

    
    from config import args
    
    filename = 'split_train_val.json'

    data_set = data_loader(args, filename, 'val', sample_slices=5)
    print('length of dataset: ', len(data_set))

    data_load = DataLoader(dataset=data_set, batch_size=1, shuffle=True, num_workers=8)

    for i, sample in enumerate(data_load):
        print("ct: {}, seg: {}, edge: {}".format(sample['image'].shape, sample['label'].shape, sample['edge'].shape))
        break