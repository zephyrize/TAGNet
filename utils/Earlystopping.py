import os
import torch
import numpy as np
from utils.file_utils import mkdir


class EarlyStopping:

    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, args=None, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, model_name='Unet', fold=1):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.args = args
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

        self.val_dice_max = -np.Inf
        self.best_dice = None

        self.delta = delta
        self.trace_func = trace_func

        self.epoch = None

        if args.model_remark == '':
            self.path = 'model/' + args.dataset + '/' + model_name + '_'  + path
        else:
            self.path = 'model/' + args.dataset + '/' + model_name + '_' + args.model_remark + '_' + path
        
        self.best_model_path = self.path.replace('checkpoint', 'withBestModel')

        self.last_model_path = self.path.replace('checkpoint', 'finalModel')

        if 'model' not in os.listdir():
            mkdir('model')
        if args.dataset not in os.listdir('model'):
            mkdir('model/' + args.dataset)
        
        if args.kfold > 0:
            if args.model not in os.listdir('model/' + args.dataset):
                mkdir('model/' + args.dataset + '/' + args.model)
            self.path = 'model/' + args.dataset + '/' + args.model + '/' + args.model + '_fold' + str(fold) + '_' + path
        
    def __call__(self, val_loss, dice, model, epoch, log):

        self.log = log
        score = -val_loss
        self.save_last_model(model, epoch)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

        if self.best_dice is None:
            self.best_dice = dice
            self.save_dice_checkpoint(dice, model)
        elif dice > self.best_dice:
            self.best_dice = dice
            self.save_dice_checkpoint(dice, model)
    

    def save_last_model(self, model, epoch):

        self.trace_func('Saving recent trained / last model ...')
        self.log.info('Saving recent trained / last model ...')

        self.save_model(model, self.last_model_path)


    def save_checkpoint(self, val_loss, model):

        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.log.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        self.save_model(model, self.path)
        
        self.val_loss_min = val_loss

    def save_dice_checkpoint(self, dice, model):

        '''Saves model when dice increase.'''

        self.trace_func(f'dice increased ({self.val_dice_max:.6f} --> {dice:.6f}).  Saving model ...')
        self.log.info(f'dice increased ({self.val_dice_max:.6f} --> {dice:.6f}).  Saving model ...')

        self.save_model(model, self.best_model_path)
        
        self.val_dice_max = dice

    def save_model(self, model, model_path):
        if len(self.args.gpu) == 1:
            torch.save(model.state_dict(), model_path)
        else:
            torch.save(model.module.state_dict(), model_path)