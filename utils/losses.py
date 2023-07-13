import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.functional import cross_entropy
from torch.nn.modules.loss import _WeightedLoss


class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):

        smooth = 1
        dice = 0.

        for class_index in range(1, pred.size(1)):
            inter = torch.sum(pred[:, class_index,...] * target[:, class_index,...])
            union = torch.sum(pred[:, class_index,...]) + torch.sum(target[:, class_index,...])
            dice += (2. * inter + smooth) / (union + smooth)
        
        dice = torch.tensor(dice)
        return torch.clamp((1 - dice).mean(), 0, 1)



class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

        self.dice = DiceLoss()
    
    def forward(self, pre, label, smooth=1):

        dice_loss = self.dice(pre, label)
        
        BCE_loss = F.binary_cross_entropy(pre, label)
        
        return 0.5*dice_loss + 0.5*BCE_loss




class LogNLLLoss(_WeightedLoss):
    __constants__ = ['weight', 'reduction', 'ignore_index']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction=None,
                 ignore_index=-100):
        super(LogNLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, y_input, y_target):

        if len(y_target.shape) == 4 and y_target.shape[1] == 1:
            y_target = y_target.squeeze()
        # y_input = torch.log(y_input + EPSILON)
        return cross_entropy(y_input, y_target.long(), weight=self.weight,
                             ignore_index=self.ignore_index)


from monai.losses import FocalLoss


class focalLoss(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.focal_loss = FocalLoss()

    
    def forward(self, pred, target):

        return self.focal_loss(pred, target)



from monai.losses import TverskyLoss


class tverskyLoss(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.tversky_loss = TverskyLoss()

    
    def forward(self, pred, target):

        return self.tversky_loss(pred, target)


from monai.losses import FocalLoss


class focalLoss(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.focal_loss = FocalLoss()

    
    def forward(self, pred, target):

        return self.focal_loss(pred, target)

class FTLOSS(nn.Module):

    def __init__(self):

        super().__init__()


    def forward(self, pred, target):

        smooth = 1
        gama = 0.75

        pred_pos = torch.flatten(pred)
        target_pos = torch.flatten(target)

        TP = torch.sum(pred_pos*target_pos)
        FN = torch.sum(target_pos*(1-pred_pos))
        FP = torch.sum((1-target_pos) * pred_pos)

        alpha = 0.25

        Tv = (TP+smooth) / (TP + alpha*FN + (1-alpha) * FP + smooth)
        
        FT = torch.pow((1-Tv), gama)

        return torch.clamp(FT.mean(), 0, 2)


import monai.losses as monai_loss
from utils.helper import get_current_consistency_weight
from config import args

class PDCLoss(nn.Module):

    def __init__(self):
        super(PDCLoss, self).__init__()

        self.dice = monai_loss.DiceLoss(include_background=False)

    def cross_entropy_loss_RCF(self, prediction, labelf, beta=1.1):
        label = labelf.long()
        mask = labelf.clone()
        num_positive = torch.sum(label==1).float()
        num_negative = torch.sum(label==0).float()

        mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
        mask[label == 0] = beta * num_positive / (num_positive + num_negative)
        mask[label == 2] = 0
        cost = F.binary_cross_entropy(
                prediction, labelf, weight=mask, reduction='sum')

        return cost

    def get_consis_loss(self, slice_prop, epoch) :
        
        false_gt = [(0.5 * prop[:,0,...] + 0.5 * prop[:,2,...]).unsqueeze(1) for prop in slice_prop]
        mid_slice = [prop[:,1,...].unsqueeze(1) for prop in slice_prop]
        auxiliary_loss = sum([self.dice(mid_slice[i], false_gt[i]) for i in range(len(false_gt))]) / len(false_gt)
        return get_current_consistency_weight(epoch) * auxiliary_loss

    def forward(self, pred, target, epoch):

        final_pred, slice_prop, edge = pred[0], pred[1], pred[2]

        dice_loss = self.dice(final_pred, target[0])
        consis_loss = self.get_consis_loss(slice_prop, epoch)
        #edge_loss = self.cross_entropy_loss_RCF(edge, target[-1])

        edge_loss = F.binary_cross_entropy(edge, target[-1])
        loss = dice_loss + consis_loss + args.edge_weight * edge_loss

        return loss




# for ATM_V9
class DiceConsisLoss(nn.Module):

    def __init__(self):
        super(DiceConsisLoss, self).__init__()

        self.dice = monai_loss.DiceLoss(include_background=False)

    def get_consis_loss(self, slice_prop, epoch) :
        
        false_gt = [(0.5 * prop[:,0,...] + 0.5 * prop[:,2,...]).unsqueeze(1) for prop in slice_prop]
        mid_slice = [prop[:,1,...].unsqueeze(1) for prop in slice_prop]
        auxiliary_loss = sum([self.dice(mid_slice[i], false_gt[i]) for i in range(len(false_gt))]) / len(false_gt)
        return get_current_consistency_weight(epoch) * auxiliary_loss
    
    def forward(self, pred, target, epoch):
        
        final_pred, slice_prop = pred[0], pred[1]
        dice_loss = self.dice(final_pred, target)
        consis_loss = self.get_consis_loss(slice_prop, epoch)

        loss = dice_loss + consis_loss

        return loss



# for TAGNet

class SATLoss(nn.Module):

    def __init__(self):
        super(SATLoss, self).__init__()

        self.dice = monai_loss.DiceLoss(include_background=False)

    def get_consis_loss(self, slice_prop, epoch) :
        
        mid_slice = [prop[0].unsqueeze(1) if len(prop[0].shape)==3 else prop[0] for prop in slice_prop]
        false_gt = [prop[1].unsqueeze(1) if len(prop[1].shape)==3 else prop[1] for prop in slice_prop]

        auxiliary_loss = sum([loss_ssim(mid_slice[i], false_gt[i], window_size=11) for i in range(len(false_gt))]) / len(false_gt)
        return get_current_consistency_weight(epoch) * auxiliary_loss
    
    def forward(self, pred, target, epoch):
        
        
        final_pred, slice_prop = pred[0], pred[1]
        dice_loss = self.dice(final_pred, target)
        consis_loss = self.get_consis_loss(slice_prop, epoch)

        loss = dice_loss + consis_loss

        return loss
    

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from math import exp


def gaussian(window_size, sigma):
    """
    :param window_size:
    :param sigma:
    :return:
    """
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / (float(2 * sigma ** 2))) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5) 
    # print(_1D_window)
    _2D_window = torch.mm(_1D_window.unsqueeze(1), _1D_window.unsqueeze(1).t()).float().unsqueeze(0).unsqueeze(0)  # 二维高斯分布的权重矩阵使用一维高斯向量称其转置,在第一维再加两个维度
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):

    mu_img1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu_img2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu_img1_sq = mu_img1.pow(2)
    mu_img2_sq = mu_img2.pow(2)

    mu1_time_mu2 = mu_img1 * mu_img2
    # sigma^2 = E(x^2)-E^2(x)
    sigma_img1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu_img1_sq
    sigma_img2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu_img2_sq
    # sigma_xy = E(xy)-E(x)E(y)
    sigma_12_sq = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_time_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_time_mu2 + C1) * (2 * sigma_12_sq + C2)) / (
            (mu_img1_sq + mu_img2_sq + C1) * sigma_img1_sq + sigma_img2_sq + C2)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(self.window_size, channel=self.channel)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

        if img1.is_cuda:
            window = window.cuda(img1.get_device())

        window = window.type_as(img1)

        self.window = window
        self.channel = channel

        # return _ssim(img1, img2, window, self.window_size, channel, self.size_average)
        return -torch.log(_ssim(img1, img2, window, self.window_size, channel, self.size_average))


def ssim(img1, img2, window_size=11, size_average=True):
    _, channel, _, _ = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def loss_ssim(img1, img2, window_size=11, size_average=True):
    loss = 1-(ssim(img1, img2, window_size, size_average))
    return loss

