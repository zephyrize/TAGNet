import os
import numpy as np
import SimpleITK as sitk


from os.path import join

from utils.file_utils import save_json, load_json
import medpy.metric.binary as metric


'''

dir:


predict
    ---Unet
        --raw
        --postprocess
    ---AttUnet
        --raw
        --postprocess
    ...
    ---Other models
        --raw
        --postprocess

    ---raw_summary.json
    ---postprocess_summary.json
'''



def check_spacing(spacing_gt, spacing_pre):

    same_spac = np.all(np.isclose(spacing_gt, spacing_pre))
    
    if not same_spac:
        print("the spacing does not match between the images")
        print(spacing_gt)
        print(spacing_pre)


def get_metric_dic(pre_path, gt_path):
    
    res_dic = {}

    for case in os.listdir(gt_path):

        case_name = case.split('_label')[0]
        
        gt = sitk.ReadImage(join(gt_path, case))
        pre = sitk.ReadImage(join(pre_path, case.replace("_label", "")))
        
        spacing = np.array(gt.GetSpacing())[[2, 1, 0]]

        # check_spacing(spacing_gt, spacing_pre)

        gt = sitk.GetArrayFromImage(gt)
        pre = sitk.GetArrayFromImage(pre)

        metric_dic = {}
        metric_dic['dice'] = metric.dc(gt, pre)
        metric_dic['jaccard'] = metric.jc(gt, pre)
        metric_dic['precision'] = metric.precision(gt, pre)
        metric_dic['recall'] = metric.recall(gt, pre)
        metric_dic['specificity'] = metric.specificity(gt, pre)
        metric_dic['hd'] = metric.hd(gt, pre, voxelspacing=spacing)
        metric_dic['hd95'] = metric.hd95(gt, pre, voxelspacing=spacing)
        metric_dic['assd'] = metric.assd(gt, pre, voxelspacing=spacing)
        metric_dic['asd'] = metric.asd(gt, pre, voxelspacing=spacing)
        
        # metric_dic['sensitivity'] = metric.sensitivity(gt, pre) # = recall
        # metric_dic['true_positive_rate'] = metric.true_positive_rate(gt, pre) # = recall
        # metric_dic['ravd'] = metric.ravd(gt, pre)
        # metric_dic['volume_correlation'] = metric.volume_correlation(gt, pre)

        res_dic[case_name] = metric_dic

        print('Case: {} calculated done...'.format(case_name))

    avg = {}

    for name, case_metric in res_dic.items():
        for key, value in case_metric.items():
            if key not in avg.keys():
                avg[key] = value
            else:
                avg[key] += value

    for key, value in avg.items():
        avg[key] = avg[key] / len(res_dic.keys())
    
    res_dic['Average'] = avg

    return res_dic


def get_summary(predict_path, gt_path, model_name):

    save_raw_dir = join(predict_path, 'raw_summary.json')
    save_postprocess_dir = join(predict_path, 'postprocess_summary.json')

    if not os.path.exists(save_raw_dir) or not os.path.exists(save_postprocess_dir):
        raw_metric_dic = {}
        postprocess_metric_dic = {}
    else:
        raw_metric_dic = load_json(save_raw_dir)
        postprocess_metric_dic = load_json(save_postprocess_dir)

    for model in os.listdir(predict_path):

        if model not in model_name:
            continue    

        model_path = join(predict_path, model)
        
        raw_path = join(model_path, 'raw')
        postprocess_path = join(model_path, 'postprocess')
        # if not os.path.exists(postprocess_path):
        #     os.mkdir(postprocess_path)
        #     print("postprocess dir maked successfully...")

        print('Begin {} raw calculation...'.format(model))
        raw_metric_dic[model] = get_metric_dic(raw_path, gt_path)

        print('Begin {} postprocess calculation...'.format(model))
        postprocess_metric_dic[model] = get_metric_dic(postprocess_path, gt_path)

        print('Model: {} calculated done...'.format(model))

    save_json(raw_metric_dic, save_raw_dir)
    save_json(postprocess_metric_dic, save_postprocess_dir)
    
    print('Summary json file has been saved done...')

if __name__ == '__main__':
    
    predict_path = '/data1/zfx/data/BileDuct/predict'
    gt_path = '/data1/zfx/data/BileDuct/raw_data/labelsTs'
    model_name = ['TAGNet']

    get_summary(predict_path, gt_path, model_name)