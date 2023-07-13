import torch.nn as nn

class InitWeights(object):
    def __init__(self, init_type, neg_slope=1e-2):
        self.neg_slope = neg_slope
        self.type = init_type

    def __call__(self, module):

        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            if self.type == 'xavier':
                module.weight = nn.init.xavier_normal_(module.weight, gain=self.neg_slope)
            elif self.type == 'kaiming':
                module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            
            if module.bias is not None:    
                module.bias = nn.init.constant_(module.bias, 0)