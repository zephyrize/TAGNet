import os
import shutil
import numpy as np
import torch,random
import pandas as pd
from os.path import join
import utils.helper as utils
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter, writer

class Train_Logger():
    def __init__(self, args, save_name='_train_log'):
        self.log = None
        self.args = args
        self.summary = None
        self.save_path = args.save_path

        if args.model_remark == '':
            prefix = args.model
        else:
            prefix = args.model + '_' + args.model_remark
            
        self.save_name = prefix + save_name
        self.runs_dir = join(args.save_path, args.dataset, args.model)

        if args.save_path not in os.listdir():
            utils.mkdir(args.save_path)
        if args.dataset not in os.listdir('runs'):
            utils.mkdir(join(args.save_path, args.dataset))
        if args.model in os.listdir(join(args.save_path, args.dataset)):
            shutil.rmtree(self.runs_dir)

    def update(self,epoch,train_log,val_log):
        item = OrderedDict({'epoch':epoch})
        item.update(train_log)
        item.update(val_log)
        # item = dict_round(item,4)
        self.update_csv(item)
        self.update_tensorboard(item)

    def update_csv(self,item):
        tmp = pd.DataFrame(item,index=[0])
        if self.log is not None:
            self.log = self.log.append(tmp, ignore_index=True)
        else:
            self.log = tmp
        self.log.to_csv('%s/%s.csv' %(self.save_path,self.save_name), index=False)

    def update_tensorboard(self,item):
        if self.summary is None:
            self.summary = SummaryWriter('%s/' % self.runs_dir)
        epoch = item['epoch']
        '''
        for key,value in item.items():
            if key != 'epoch': self.summary.add_scalar(key, value, epoch)
        '''
        loss_dict = {'Train_Loss': item['Train_Loss'], 'Val_Loss': item['Val_Loss']}

        self.summary.add_scalars(self.args.model + '_Loss', loss_dict, epoch)
        self.summary.add_scalar(self.args.model + '_Performance/dice', item['Val_dice_vessel'], epoch)
