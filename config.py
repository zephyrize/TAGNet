import argparse


parser = argparse.ArgumentParser(description='set hyper parameters.')

# hyper parameters.
parser.add_argument('--n_classes', type=int, default=1, help="classes number")
parser.add_argument('--epoch', type=int, default=500, help='epoch: default = 100')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate: default = 0.001')
parser.add_argument('--batch_size', type=int, default=8, help='Unet:32; Unet++:16; AttUnet:?')
parser.add_argument('--sample_frequency', type=int, default=8, help='sample frequency')
parser.add_argument('--loss_func', type=str, default='dice', help='Defalut: dice; dice/diceBCE/focalBCE/')
parser.add_argument('--model', type=str, default='Unet', help='Defalut: Unet; Unet/AttUnet/NestedUnet/Unet_PAM/')

# parser.add_argument('--model_path_saved', type=str, default='')

# Preprocess parameters
parser.add_argument('--crop_slices', type=int, default=48)
parser.add_argument('--crop_image_size', type=int, default=128)
parser.add_argument('--val_crop_max_size', type=int, default=96)
parser.add_argument('--test_rate', type=float, default=0.2, help='')
parser.add_argument('--valid_rate', type=float, default=0.2, help='')
parser.add_argument('--xy_down_scale', type=float, default=0.5, help='')
parser.add_argument('--norm_factor', type=float, default=200.0, help='')
parser.add_argument('--slice_down_scale', type=float, default=1.0, help='')
parser.add_argument('--scale', type=int, default=2, help='AATM_V6 scale')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--edge_weight', type=float, default=0.1, help='edge_weight')

parser.add_argument('--save_path',default = 'runs',help='tensorboard saved path')
parser.add_argument('--data_root_dir', type=str, default='/data1/zfx/data/', help='data_root_dir: default = /home/sophia/zfx/data/')

parser.add_argument('--kfold', type=int, default=0, help='Cross-validation: default = 0')
parser.add_argument('--model_remark', type=str, default="", help='This is a remark for model')
parser.add_argument('--dataset', type=str, default='BileDuct', help='Defalut: ZDYY; 3Dircadb/ZDYY/bileDuct')
parser.add_argument('--patience', type=int, default=50, help='Early stopping number: default = 12')
# parser.add_argument('--device_ids', type=int, nargs='+', default=[4,5,6,7], help='Multiple GPUs number: defalut = [5,6,7]')

parser.add_argument('--gated_scale', type=float, default=0.125, help='gated attention scale: default = 0.125')
parser.add_argument('--best_model', default=False, action='store_true', help='choose the best model')
parser.add_argument('--continue_training', default=False, action='store_true', help='whether to continue training')
parser.add_argument('--begin_epoch', type=int, default=1, help='epoch of begin training')
parser.add_argument('--final_model', default=False, action='store_true', help='choose final model')
parser.add_argument('--dsv', default=False, action='store_true', help='deeep supervision')
parser.add_argument('--save_infer', default=False, action='store_true', help='save predicted map or not')
parser.add_argument('--use_cpu', default=False, action='store_true', help='whether to use gpu for testing')
parser.add_argument('--mulGPU', default=False, action='store_true', help='whether to use multiple GPUSs')
parser.add_argument('-gpu', '--gpu', nargs='+', default='None', help="GPU ids to use")
parser.add_argument('--postprocess', default=False, action='store_true', help='whether to run postprocessing')
parser.add_argument('--save_csv', default=False, action='store_true', help='save predicted map or not')


# for cropping and preprocessing data
parser.add_argument('--sample_slices', type=int, default=3, help='consecutive slices.')
parser.add_argument('--process_type', type=str, default='2D', help='preprocessing type: 2D: interpolation; 2.5D: z-axis interpolation; 3D: all-axis interpolation')
parser.add_argument('--process_mode', type=str, default='train', help='crop and preprocess mode: train or val')


args = parser.parse_args()
