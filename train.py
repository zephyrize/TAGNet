import os
import config
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in config.args.gpu])

import time
import torch
import torch 
import logging

from tqdm import tqdm
from os.path import join
from rich.progress import track
from collections import OrderedDict

import torch.optim as optim
import utils.logger as logger
import utils.metrics as metrics

from utils.build_model import *
from utils.Earlystopping import *
from utils.helper import get_device
from utils.helper import test_single_volume
from utils.torch_poly_lr_decay import PolynomialLRDecay

from datasets.data_loader import *

device = get_device()

if 'logout' not in os.listdir():
    os.mkdir('logout')
logging.basicConfig(filename = join('logout', args.model + '_' + args.model_remark +'.log'),
                    level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

log = logging.getLogger(name="logger")


def train(args, model, train_loader, optimizer, criterion, epoch):

    # print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))

    model.train()
    start_time = time.time()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_classes=args.n_classes)

    for idx, sample in track(enumerate(train_loader), total=len(train_loader), description='training'):

        image, label = sample['image'], sample['label']
        image, label = image.to(device), label.to(device)

        output = model(image)

        loss = 0

        if args.loss_func == 'pdc':
            loss = criterion(output, [label, sample['edge'].to(device)], epoch)
        elif args.loss_func == 'DiceConsis' or args.loss_func == 'sat':
            loss = criterion(output, label, epoch)
        else:
            loss = criterion(output, label)
        
        optimizer.zero_grad()

        # mixed precision training
        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()
        
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), image.size(0))

        predict = output[0] if isinstance(output, list) else output
        
        output = torch.round(predict)

        train_dice.update(predict, label)
        
    train_log = OrderedDict({'Train_Loss': train_loss.avg, 'Train_dice_vessel': train_dice.avg[0]})

    time_elapsed = time.time() - start_time

    log.info('Train time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Train time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    return train_log

def val(args, model, val_loader, criterion, epoch):

    start_time = time.time()
    
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_classes=args.n_classes)
    
    for idx, sample in track(enumerate(val_loader),total=len(val_loader), description='validation'):
        
        image, label = sample['image'], sample['label']

        PR = test_single_volume(image[0], model, args.batch_size * len(args.gpu), device, dsv=args.dsv, multi_loss=(args.model=='ATM_V9')).to(torch.float32)
        GT = label[0].to(torch.float32)
        
        assert len(PR.shape) == len(GT.shape) and PR.shape == GT.shape

        loss = criterion(PR, GT)
        val_loss.update(loss.item())

        PR = torch.round(PR)

        val_dice.update(PR, GT)

        log.info('val dice per case: {}'.format(val_dice.get_dices(PR, GT)))
        print('val dice per case: {}'.format(val_dice.get_dices(PR, GT)))

    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_dice_vessel': val_dice.avg[0]})

    time_elapsed = time.time() - start_time
    
    log.info("Val time: {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print('Val time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return val_log


if __name__ == '__main__':

    args = config.args
    
    train_loader, val_loader = get_data_loader(args)
    
    model = get_model(args, mode='train', device=device, device_ids=[i for i in range(len(args.gpu))])
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001)

    # 加入混合精度训练
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    
    scheduler = PolynomialLRDecay(optimizer, max_decay_steps=args.epoch, end_learning_rate=0.0, power=0.9)
    # scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = get_criterion(args, device=device)
    early_stopping = EarlyStopping(args=args, patience=args.patience, verbose=True, model_name=args.model)

    train_logger = logger.Train_Logger(args, "_train_log")
    
    for epoch in tqdm(range(args.begin_epoch, args.epoch+1)):
        
        train_log = train(args, model, train_loader, optimizer, criterion, epoch)
        val_log = val(args, model, val_loader, criterion, epoch)
        train_logger.update(epoch, train_log, val_log)
        
        if scheduler is not None:
            scheduler.step()
        
        if args.sample_slices >3 and epoch == 50:
            for param in model.parameters():
                param.requires_grad =True
        
        # print("Epoch: [{}]\nTrain: {} \nValid: {}".format(epoch, train_log, val_log))
        log.info("\nEpoch: [{}]\nTrain: {} \nValid: {} \nlearning rate:{} ".format\
            (epoch, train_log, val_log, optimizer.state_dict()['param_groups'][0]['lr']))

        print("\nEpoch: [{}]\nTrain: {} \nValid: {} \nlearning rate:{} ".format\
            (epoch, train_log, val_log, optimizer.state_dict()['param_groups'][0]['lr']))

        early_stopping(val_log['Val_Loss'], val_log['Val_dice_vessel'], model, epoch, log)

        log.info('\n' * 3)
    
    print('\n' * 5)


