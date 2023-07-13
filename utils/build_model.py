from ast import arg
from fileinput import filename
import os
from os.path import join
from networks.model import get_model

import utils.losses as losses
from utils.helper import load_file_name_list
from torch.utils.data import DataLoader

from datasets.data_loader import data_loader
from datasets.test_loader import test_loader

from monai.losses import DiceLoss
from monai.losses import GeneralizedDiceLoss
from monai.losses import DiceCELoss
from monai.losses import DiceFocalLoss
from monai.losses import FocalLoss
from monai.losses import TverskyLoss
import torch.nn.functional as F

import torch

def get_criterion(args, device=None):

    if args.loss_func == 'dice':
        criterion = DiceLoss(include_background=False).to(device)
    if args.loss_func == 'sat':
        criterion = losses.SATLoss().to(device)
    if args.loss_func == 'generalDice':
        criterion = GeneralizedDiceLoss(include_background=False).to(device)
    elif args.loss_func == 'diceCE':
        criterion = DiceCELoss(include_background=False).to(device)
    elif args.loss_func == 'diceFocal':
        criterion = DiceFocalLoss(include_background=False).to(device)
    elif args.loss_func == 'lognll':
        criterion = losses.LogNLLLoss().to(device)
    elif args.loss_func == 'focal':
        criterion = FocalLoss(include_background=False).to(device)
    elif args.loss_func == 'tversky':
        criterion = TverskyLoss(include_background=False).to(device)
    elif args.loss_func == 'ftloss':
        criterion = losses.FTLOSS().to(device)
    elif args.loss_func == 'pdc':
        criterion = losses.PDCLoss().to(device)
    
    return criterion

def get_data_loader(args):

    file_name = 'split_train_val.json'
    train_set = data_loader(args, file_name, mode='train', process_type=args.process_type, sample_slices=args.sample_slices)
    val_set = data_loader(args, file_name, mode='val', process_type=args.process_type, sample_slices=args.sample_slices)
    
    print('length of batch sampler: ', len(train_set))
    
    train_load = DataLoader(dataset=train_set, batch_size=args.batch_size * len(args.gpu), shuffle=True, num_workers=8, drop_last=True)
    val_load = DataLoader(dataset=val_set, batch_size=1, shuffle=False, num_workers=8)
    # logging.info('\nload data done...')
    return train_load, val_load


def get_model(args, mode='train', device=None, device_ids=None):
    
    img_ch = args.sample_slices

    if args.model == 'TAGNet':
        model =  get_model(img_size=256, dsv=args.dsv, sample_slices=args.sample_slices)

    if len(device_ids) > 1:
        print('model -> nn.DataParallel')
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        model = model.cuda()
    else:
        model = model.to(device)

    if args.continue_training is True:
        if args.model_remark != '':
            args.model_remark = '_' + args.model_remark
            model_path = join('model', args.dataset, args.model + args.model_remark + '_finalModel.pt')
            model.load_state_dict(torch.load(model_path, map_location=device))
        return model
        

    if mode == 'test':
        if args.model_remark != '':
            args.model_remark = '_' + args.model_remark

        if args.best_model is True:
            model_path = join('model', args.dataset, args.model + args.model_remark + '_withBestModel.pt')
        elif args.final_model is True:
            model_path = join('model', args.dataset, args.model + args.model_remark + '_finalModel.pt')
        else:
            model_path = join('model', args.dataset, args.model + args.model_remark + '_checkpoint.pt')

        if args.use_cpu is True:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # 'cuda:0'
        else:
            model.load_state_dict(torch.load(model_path, map_location=device))
        print("model path: ", model_path)
    
    return model