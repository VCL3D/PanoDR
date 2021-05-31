import torch
import torch.nn as nn
import torchvision.models as models
import glob
import time
import os

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def getCheckpoint(folder_path):
    
    files_D = glob.glob(folder_path.replace("*.pth", "*D*.pth"))
    files_G = glob.glob(folder_path.replace("*.pth", "*G*.pth"))
    file_times_D = list(map(lambda x: time.ctime(os.path.getctime(x)), files_D))
    file_times_G = list(map(lambda x: time.ctime(os.path.getctime(x)), files_G))
    files_D[sorted(range(len(file_times_D)), key=lambda x: file_times_D[x])[-1]]
    files_G[sorted(range(len(file_times_G)), key=lambda x: file_times_G[x])[-1]]

    return [files_D[0], files_G[0]]
