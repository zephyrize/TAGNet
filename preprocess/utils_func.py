import numpy as np

def get_nonzero_box(mask):
    mask_voxel_coords = np.where(mask != 0)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1])) 
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]
    
def compute_stats(voxels):
        
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


def extend_slices(box, extend_num, max_slices):

    if box[0][0] - extend_num >= 0:
        box[0][0] -= extend_num
    else:
        box[0][0] = 0
    
    if box[0][1] + extend_num < max_slices:
        box[0][1] += extend_num
    else:
        box[0][1] = max_slices - 1
    return box

def crop(data, liver_box):
    
    new_data = np.array([
        data[i][liver_box[0][0]:liver_box[0][1], liver_box[1][0]:liver_box[1][1], liver_box[2][0]:liver_box[2][1]].copy() \
        for i in range(data.shape[0])
    ])

    return new_data

def trans25D(data, properties, slices):

    assert slices % 2 == 1 and len(data.shape) == 4
    slices_num = data.shape[1]

    begin_slice = extend_slices = slices // 2
    end_slice = slices_num - extend_slices

    transform = lambda x : np.array([x[i-extend_slices:i+extend_slices+1].copy() for i in range(begin_slice, end_slice)])

    new_data = np.array([transform(data[i]) for i in range(data.shape[0])])

    assert len(new_data.shape) == 5

    if properties is not None:
        properties['consecutive_slices'] = slices

    return new_data, properties
    
