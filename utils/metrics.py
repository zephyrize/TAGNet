from logging import log
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class LossAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)
        # print(self.val)


class DiceAverage(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,n_classes=1):

        '''
        dice
        '''
        self.n_classes = n_classes
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.n_classes, dtype='float64')
        self.avg = np.asarray([0]*self.n_classes, dtype='float64')
        self.sum = np.asarray([0]*self.n_classes, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        
        if len(logits.shape) == len(targets.shape):
            if len(logits.shape) == 3:
                logits = torch.unsqueeze(logits, dim=1)
                targets = torch.unsqueeze(targets, dim=1)
                
        if len(logits.shape) == 3 and len(targets.shape) == 4:
            logits = torch.unsqueeze(logits, dim=1)

        if len(logits.shape) == 4 and len(targets.shape) == 3:
            targets = torch.unsqueeze(targets, dim=1)

        assert len(logits.shape) == len(targets.shape)
        
        self.value = Metricx.get_dices(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)
    

    @staticmethod
    def get_dices(logits, targets):
        dices = []
        for class_index in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_index,...] * targets[:, class_index,...])
            union = torch.sum(logits[:, class_index,...]) + torch.sum(targets[:, class_index,...])
            dice = (2. * inter + 1) / (union + 1)
            dices.append(dice.item())
        return np.asarray(dices)


class Metricx(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self,n_metrics=4):

        '''
        precision
        sensitivity/recall
        specificity
        '''
        self.n_metrics = n_metrics
        self.reset()

    def reset(self):
        self.value = np.asarray([0]*self.n_metrics, dtype='float64')
        self.avg = np.asarray([0]*self.n_metrics, dtype='float64')
        self.sum = np.asarray([0]*self.n_metrics, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        
        if type(logits) is np.ndarray:
            logits = torch.from_numpy(logits.astype(np.int16))
        if type(targets) is np.ndarray:
            targets = torch.from_numpy(targets.astype(np.int16))
            
        if len(logits.shape) == len(targets.shape):
            if len(logits.shape) == 3:
                logits = torch.unsqueeze(logits, dim=1)
                targets = torch.unsqueeze(targets, dim=1)
        if len(logits.shape) == 3 and len(targets.shape) == 4:
            logits = torch.unsqueeze(logits, dim=1)
        if len(logits.shape) == 4 and len(targets.shape) == 3:
            targets = torch.unsqueeze(targets, dim=1)


        assert len(logits.shape) == len(targets.shape)
        
        self.value[0] = Metricx.get_dices(logits, targets)
        self.value[1] = Metricx.get_precision(logits, targets)
        self.value[2] = Metricx.get_sensitivity(logits, targets)
        self.value[3] = Metricx.get_specificity(logits, targets)
        
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)
    

    @staticmethod
    def get_dices(logits, targets):
        dices = []
        for class_index in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_index,...] * targets[:, class_index,...])
            union = torch.sum(logits[:, class_index,...]) + torch.sum(targets[:, class_index,...])
            dice = (2. * inter + 1) / (union + 1)
            dices.append(dice.item())
        return np.asarray(dices)

    @staticmethod
    def get_precision(logits, targets):
        smooth = 1
        precision = []
        pre = (torch.sum(torch.multiply(logits, targets)[:,0:,...])+smooth)/(torch.sum(logits[:,0:,...])+smooth)
        precision.append(pre.item())
        return np.asarray(precision)

    @staticmethod
    def get_sensitivity(logits, targets):
        smooth = 1
        sensitivity = []
        sens = (torch.sum(torch.multiply(logits, targets)[:,0:,...])+smooth)/(torch.sum(targets[:,0:,...])+smooth)
        sensitivity.append(sens.item())
        return np.asarray(sensitivity)

    @staticmethod
    def get_sensitivity_V2(logits, targets, threshold=0.5):
        
        '''
        TP / (TP + FN)
        '''
        recall = []
        
        logits = logits > threshold
        targets = targets == torch.max(targets)

        TP = torch.sum(logits & targets)
        FN = torch.sum((logits==False) & (targets==True))

        rec = TP/(TP + FN +1e-6)
        recall.append(rec.item())

        return np.asarray(recall)
    
    @staticmethod
    def get_specificity(logits, targets, threshold=0.5):

        specificity = []

        logits = logits > threshold
        targets = targets == torch.max(targets)

        TN = torch.sum((logits==False) & (targets==False))
        FP = torch.sum((logits==True) & (targets==False))

        spec = TN/(TN + FP +1e-6)
        specificity.append(spec.item())

        return np.asarray(specificity)
