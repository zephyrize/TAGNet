import os
import config
from os.path import join
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in config.args.gpu])

from preprocessor import Preprocessor

from utils.build_model import get_model
from utils.helper import test_single_volume, get_device
from utils.file_utils import load_json, load_pickle, mkdir


from skimage import morphology
from scipy.ndimage import zoom
from rich.progress import track
from skimage.transform import resize 

import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
import medpy.metric.binary as metric
from networks.ATM_V9 import get_Unet_ATM_V9
from networks.CAT_Net import get_cat_net

device = get_device()

class inference:

    def __init__(self, args, root_dir, dataset='BileDuct', process_type='2D'):

        self.args = args
        self.type = process_type
        self.root_dir = root_dir
        self.dataset = dataset

        self.dataset_root_dir = join(root_dir, dataset)

        self.predict_raw_dir = join(self.dataset_root_dir, 'predict', args.model + '_' + args.model_remark, 'raw')
        self.predict_postprocess_dir = join(self.dataset_root_dir, 'predict', args.model + '_' + args.model_remark, 'postprocess')
        
        self.preprocessed_dir = join(self.dataset_root_dir, 'preprocessed_data')
        self.preprocess_dataset_json = join(self.preprocessed_dir, 'preprocess_dataset.json')

        self.intensityproperties_file = join(self.dataset_root_dir, 'cropped_data', 'intensityproperties.pkl')

        
        self.model = None
        self.metric = None
        self.patch_size = 256
        self.sample_slices = args.sample_slices
        self.resize_ct = lambda data, x, y: zoom(data, (1.0, self.patch_size / x, self.patch_size / y), order=3)

        self.restore_label = lambda label, x, y: zoom(label, (1.0, x / self.patch_size, y / self.patch_size), order=0)

        self.reprocess_test_data = False

        self.row_name = ['dice', 'precision', 'sensitivity']

        self.eval_matrix = 'eval_' + args.dataset + '_wo_postprocess.csv'
    
    
    def postprocess(self, pre, threshhold=320):
        
        post_arr = morphology.remove_small_objects(pre.astype(np.bool8), threshhold, connectivity=3)

        return post_arr.astype(np.float32)


    def get_image_from_array(self, arr, properties):
        
        image = sitk.GetImageFromArray(arr)
        image.SetOrigin(properties['itk_origin'])
        image.SetDirection(properties['itk_direction'])
        image.SetSpacing(properties['itk_spacing'])

        return image
    
    def save_predict(self, arr, case_name, properties):
        
        label = self.get_image_from_array(arr, properties)
        sitk.WriteImage(label, join(self.predict_raw_dir, case_name+'.nii.gz'))

        if self.args.postprocess is True:
            if not os.path.exists(self.predict_postprocess_dir):
                mkdir(self.predict_postprocess_dir)
            post_arr = self.postprocess(arr)
            post_pre = self.get_image_from_array(post_arr, properties)
            sitk.WriteImage(post_pre, join(self.predict_postprocess_dir, case_name+'.nii.gz'))
        
        print('Inference of case {} has been saved done...\n\n'.format(case_name))


    def preprocess_data(self):
        
        preprocess_json = load_json(self.preprocess_dataset_json)

        if self.reprocess_test_data is False and \
            "test" in preprocess_json.keys() and \
                len(preprocess_json['test']) > 0 and \
                    "preprocess_npy" in preprocess_json['test'][0].keys():
            print('Test data have been preprocessed before')
        else:
            print('Test data have not been preprocessed yet')
            
            preprocesor = Preprocessor(self.root_dir, self.dataset, mode='test')
            preprocesor.crop_data()
            preprocesor.preprocess_data()
            
            print('Test data have been preprocessed done')
    
    def pad2origin(self, arr, box, origin_shape, extend_slices=0):

        pad_arr = np.pad(arr, ((box[0][0]+extend_slices, origin_shape[0]-box[0][1]+extend_slices), 
                                (box[1][0], origin_shape[1]-box[1][1]), 
                                (box[2][0], origin_shape[2]-box[2][1])), 'constant', constant_values=(0, 0))
        assert pad_arr.shape[0] == origin_shape[0] and pad_arr.shape[1] == origin_shape[1] and pad_arr.shape[2] == origin_shape[2]

        return pad_arr

    def reduction_for2D(self, predict, properties):
        
        predict = self.restore_label(predict, properties['crop_shape'][1], properties['crop_shape'][2])
        predict = self.pad2origin(predict, properties['liver_box'], properties['origin_shape'], 0)

        return predict
    
    def reduction_for25D(self, predict, properties):

        '''
        there exists a bug in the function that is not fixed.
        '''

        extend_slices = properties['consecutive_slices'] // 2
        predict = self.restore_label(predict, properties['new_crop_shape'][1], properties['new_crop_shape'][2])
        predict = self.pad2origin(predict, properties['new_box'], properties['interpolation_shape'], extend_slices)
        
        scale_rate = properties['new_spacing'] / properties['spacing']
        new_shape = np.round(predict.shape * scale_rate).astype(np.int)
        
        reduction_pre = resize(predict.astype(np.float32), new_shape, order=0).astype(np.uint8)
        reduction_pre = self.pad2origin(reduction_pre, properties['origin_box'], properties['origin_shape'], 0)

        return reduction_pre

    def reduction(self, predict, properties):
                
        if self.type == '2D':
            predict = self.reduction_for2D(predict, properties)
        if self.type == '2.5D':
            predict = self.reduction_for25D(predict, properties)
        if self.type == '3D':
            pass

        return predict
        
    def get_attribute(self, case_dic):

        name = case_dic['name']
        data = np.load(case_dic['preprocess_npy']) # shape: (3, all_slices, consecutive slices, h, w)
        properties = load_pickle(case_dic['preprocess_pkl'])
        GT = sitk.GetArrayFromImage(sitk.ReadImage(case_dic['label']))

        # 计算取样的个数: 3/5/7
        self.consecutive_slices = data.shape[2]
        mid_slice = self.consecutive_slices // 2
        image = data[0][:, mid_slice-self.sample_slices//2:mid_slice+self.sample_slices//2+1,...].copy()

        return name, image, GT, properties


    def cal_metric(self, PR, GT):
        
        assert len(PR.shape) == 3 and len(PR.shape) == len(GT.shape)

        res = {}

        res['Dice'] = metric.dc(PR, GT)
        res['Jaccard'] = metric.jc(PR, GT)
        res['Precision'] = metric.precision(PR, GT)
        res['Recall'] = metric.recall(PR, GT)
        # res['Specificity'] = metric.specificity(PR, GT)

        return res


    def run(self, model=None):
        
        self.preprocess_data()

        self.test_data = load_json(self.preprocess_dataset_json)['test']
        
        self.model = model
        if self.model is None:
            self.model = get_model(self.args, mode='test', device=device, device_ids=[i for i in range(len(self.args.gpu))])

        self.metric_res = {}

        for idx, case_dic in track(enumerate(self.test_data),total=len(self.test_data), description='test'):
            
            case_name, image, GT, properties = self.get_attribute(case_dic)

            PR = test_single_volume(image, self.model, self.args.batch_size, device=device, dsv=self.args.dsv, use_cpu=self.args.use_cpu, multi_loss=(self.args.model=='ATM_V9')).to(torch.float32)
            PR = torch.round(PR).squeeze().numpy() # (z, x, y)  value : (0 or 1)
            
            restore_PR = self.reduction(PR, properties)
            
            self.metric_res[case_name] = self.cal_metric(restore_PR, GT)

            if self.args.save_infer is True:
                if not os.path.exists(self.predict_raw_dir):
                    mkdir(self.predict_raw_dir)
                
                self.save_predict(restore_PR, case_name, properties)

        self.print_metrics()

        if self.args.save_csv is True:
            self._res2csv()
    

    def print_metrics(self):

        self.avg = {}        
        for name, case_metric in self.metric_res.items():
            print('Case-name: ', name)
            for key, value in case_metric.items():
                print('{}: {}'.format(key, value))
                if key not in self.avg.keys():
                    self.avg[key] = value
                else:
                    self.avg[key] += value
            print('\n')

        print('AVG:')
        for key, value in self.avg.items():
            print('{}: {}'.format(key, value / len(self.metric_res.keys())))
            self.avg[key] = value / len(self.metric_res.keys())
        print('\n')

        self.metric_res['AVG'] = self.avg

    def get_avg(self):
        
        return self.avg

    def _res2csv(self):
        
        eval_matrix = None
        
        row_tuple = []
        data = []
        for key in self.metric_res.keys():
            model_name = self.get_model_name(self.args.model)
            row_tuple.append((model_name, key))
            data.append(list(self.metric_res[key].values()))

        columns = list(self.metric_res['AVG'].keys())
        

        if self.eval_matrix not in os.listdir():
            
            eval_matrix = pd.DataFrame(
                            data=np.array(data),
                            index = pd.MultiIndex.from_tuples(row_tuple),
                            columns = columns
                        )
            eval_matrix.describe().round(4)
            eval_matrix.index.names = ['Model', 'Case']

        else:
            print('add {} test result to eval_matrix.csv file'.format(self.args.model))
            eval_matrix = pd.read_csv(self.eval_matrix).set_index(['Model', 'Case'])
            add_frame = pd.DataFrame(
                        data=np.array(data),
                        index = pd.MultiIndex.from_tuples(row_tuple),
                        columns = columns
            )

            eval_matrix = eval_matrix.append(add_frame)
            eval_matrix.describe().round(4)
            eval_matrix = eval_matrix.groupby(['Model', 'Case']).mean()
        
        eval_matrix.to_csv(self.eval_matrix)

        print('test out have been saved to eval_matrix.csv file')

    def get_model_name(self, model_name):
        if model_name == 'Unet' and self.sample_slices == 1:
            return 'Unet-2D'
        if model_name == 'Unet' and self.sample_slices == 3:
            return 'Unet-2.5D'
        if model_name == 'NestedUnet':
            return 'Unet++'
        if model_name == 'offical_transUnet':
            return 'transUnet'
        if model_name ==' MNet':
            return 'Mnet'
        
        if model_name == 'UnetASPP':
            return 'Base+ASPP'
        
        
        return model_name

if __name__ == '__main__':

    from config import args
    
    root_dir = args.data_root_dir
    dataset = args.dataset
    process_type = args.process_type

    print('Test information: ')
    
    infer = inference(args, root_dir, dataset=dataset, process_type=process_type)
    infer.run()

