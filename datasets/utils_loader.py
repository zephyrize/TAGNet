import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndimage

def preprocess(self, file):

    # print(file)
    npy_path, pkl_path = file[0], file[1]

    data = np.load(npy_path)

    # properties = pickle.load(open(pkl_path, 'rb'))
    # box = properties['new_box']
    # crop_data = data[:, box[0][0]:box[0][1],...]

    ct, seg = data[0], data[1]

    seg[seg>0] = 1

    # resize to 256 * 256
    
    percentile_99_5 = np.percentile(self.voxels_all, 99.5)
    percentile_00_5 = np.percentile(self.voxels_all, 00.5)

    ct = np.clip(ct, percentile_00_5, percentile_99_5)
    ct = (ct - self.voxels_mean) / self.voxels_std
    ct = ct.astype(np.float32)
    
    return ct, seg


def load_file_name_list(file_path):
    
    file_name_list = []
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline().strip()  # 整行读取数据
            if not lines:
                break
            file_name_list.append(lines.split())
    return file_name_list

def get_voxels_all(train_path_lists):
        
    res_voxels = []

    for file in train_path_lists:

        npy_file = file[0]
        data = np.load(npy_file)
        
        temp_mask = data[1] > 0
        voxels = list(data[0][temp_mask][::10])

        res_voxels = res_voxels + voxels

    return res_voxels



def crop_img(img, crop_size):

    slice, h, w = img.shape
    if h >= crop_size and w >= crop_size:
        crop_h = h // 2 - (crop_size // 2)
        crop_w = w // 2 - (crop_size // 2)
        return img[:, crop_h:(crop_h + crop_size), crop_w: (crop_w + crop_size)]
    elif h < crop_size:
        if w >= crop_size:
            pad_h = crop_size // 2 - h // 2
            crop_w = w // 2 - (crop_size // 2)
            temp = np.zeros((slice, crop_size, crop_size))
            temp[:, pad_h:pad_h + h, :] = img[:, :, crop_w:crop_w + crop_size]
            return temp
        else:
            pad_h = crop_size // 2 - h // 2
            pad_w = crop_size // 2 - w // 2
            temp = np.zeros((slice, crop_size, crop_size))
            temp[:, pad_h:pad_h + h, pad_w: pad_w + w] = img
            return temp
    elif w < crop_size:
        crop_h = h // 2 - (crop_size // 2)
        pad_w = crop_size // 2 - w // 2
        temp = np.zeros((slice, crop_size, crop_size))
        temp[:, :, pad_w:pad_w + w] = img[:, crop_h:crop_h+crop_size, :]
        return temp


import torch
from torch.utils.data import Sampler

class RandomSampler(Sampler):

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(torch.randperm(len(self.data_source)).long())

    def __len__(self):
        return self.data_source.__len__()



class RandomSampler(Sampler):

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(torch.randperm(len(self.data_source)).long())

    def __len__(self):
        return self.data_source.__len__()


class SequentialSampler(Sampler):

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return self.data_source.__len__()


class BatchSampler(object):
    def __init__(self, sampler, batch_size, drop_last=False, sample_frequency=8):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.sample_frequency = sample_frequency

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            for i in range(self.sample_frequency):
                batch.append(idx)
                yield batch
                batch = [] 
            
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        # if self.drop_last:
        #     return (len(self.sampler) // self.batch_size) * self.sampler_frequency
        # else:
        #     return ((len(self.sampler) + self.batch_size - 1) // self.batch_size)* self.sampler_frequency
        return len(self.sampler) * self.sample_frequency


from config import args

import random
import torchvision.transforms.functional as tf

def data_augmentation(image, label):

    if random.random() > 0.5:
        image = tf.hflip(image)
        label = tf.hflip(label)
    if random.random() > 0.5:
        image = tf.vflip(image)
        label = tf.vflip(label)
    if random.random() > 0.5:
        angle = random.randint(-30, 30)
        image = tf.rotate(image, angle)
        label = tf.rotate(label, angle)
    
    return image, label

def my_collate(batch, batch_size=args.batch_size):


    volume, seg = zip(*batch)

    assert len(volume) == 1

    # sample
    slice_num = volume[0].shape[0]
    random_slice = np.random.randint(batch_size, slice_num - batch_size)

    batch_image = volume[0][random_slice-batch_size//2 : random_slice+batch_size//2,...]
    batch_label = seg[0][random_slice-batch_size//2 : random_slice+batch_size//2,...]

    batch_image, batch_label = torch.from_numpy(batch_image), torch.from_numpy(batch_label)
    
    batch_image, batch_label = data_augmentation(batch_image, batch_label)
    
    return batch_image.unsqueeze(1), batch_label.unsqueeze(1)



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, maskresize=None, imageresize=None):
        
        image, label, edge = sample['image'], sample['label'], sample['edge']
        if len(label.shape) == 2:
            label = label.reshape((1,)+label.shape)
        if len(edge.shape) == 2:
            edge = edge.reshape((1,)+edge.shape)
        if len(image.shape) == 2:
            image = image.reshape((1,)+image.shape)
        return {'image': torch.from_numpy(image).float(),
                'label': torch.from_numpy(label).float(),
                'edge': torch.from_numpy(edge).float()}