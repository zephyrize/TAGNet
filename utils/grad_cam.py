import sys
sys.path.append('../')

import os
from os.path import join
import torch
from monai.visualize import GradCAM
from networks.Unet import *
from networks.AATM import get_Unet_AATM
from utils.file_utils import load_json
from utils.helper import show_img

import numpy as np
import cv2
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      save_img: bool = False,
                      save_path: str = None,
                      vis_layer: str = None,
                      save_pic_index: int = 0 ) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """

    img_arr = None

    if np.max(img) > 1:
        # print("The input image should np.float32 in the range [0, 1] \n Apply max-min to normalize to range [0, 1]")
        img = (img - img.min()) / (img.max() - img.min())
    
    if len(img.shape) == 2:
        img_arr = np.empty([256, 256, 3])
        img_arr[...] = np.expand_dims(img, axis=-1).astype(np.float32)


    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    cam = heatmap + img_arr
    cam = cam / np.max(cam)

    if save_img is True:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        cv2.imwrite(join(save_path, str(save_pic_index).zfill(3) + '_img_{}.png'.format(vis_layer)), np.uint8(255 * img_arr))
        cv2.imwrite(join(save_path, str(save_pic_index).zfill(3) + '_cam_{}.png'.format(vis_layer)), np.uint8(255 * cam))

    return np.uint8(255 * cam)

def get_data(index):

    '''
    param index: 0: case_002, 1: case_010, 2: case_012, 3: case_013
    '''
    data_json = '/home/sophia/zfx/data/BileDuct/preprocessed_data/preprocess_dataset.json'
    dataset = load_json(data_json)
    test_data = dataset['test'][index]['preprocess_npy']

    data = np.load(test_data)

    return data[0]


def get_model(model_name):

    model_path = '/home/sophia/zfx/code/segBileDuct/model/BileDuct/' + model_name + '_dice_channel_256_withBestModel.pt'
    model = Unet(3, 1)
    # model = get_Unet_AATM(img_size=256)
    model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    model.eval()
    return model

if __name__ == '__main__':

    case_index = 2
    slice_index = 60


    case_dic = {
        0 : 'BileDuct_002',
        1 : 'BileDuct_010',
        2 : 'BileDuct_012',
        3 : 'BileDuct_013',
    }

    image = get_data(case_index)# shape:(z, 3, x, y)
    model_name = 'Unet'
    model = get_model(model_name)

    # vis_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']

    data = torch.from_numpy(image.astype(np.float32))
    length = data.shape[0]

    # heatmap = torch.tensor([])
    cam = GradCAM(nn_module=model, target_layers='conv1_')
    img = data[slice_index].unsqueeze(0)
    res = cam(x=img)
    # heatmap = torch.cat([heatmap, res], dim=0)
    vis = show_cam_on_image(img.squeeze()[1].numpy(), 
                            mask=res[0][0], 
                            use_rgb=True,
                            save_img=False)
                            # save_path=join('../cam_pic','Unet', case_dic[case_index]),
                            # vis_layer = vis_layers[i],
                            # save_pic_index = slice + 1)
    plt.imshow(vis)
    plt.show()
    # print(vis_layers[i] + ' level have been vised done...')

