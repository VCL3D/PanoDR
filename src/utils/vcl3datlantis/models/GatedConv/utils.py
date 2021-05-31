import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision as tv
from vcl3datlantis.models.GatedConv import network
#import dataset

# ----------------------------------------
#                 Network
# ----------------------------------------
def create_RegressionGenerator(opt):

        # Initialize the networks
    generator = network.GatedRegressionGenerator(opt)
    print('Generator is created!')
    if opt.load_name:
        generator = load_dict(generator, opt.load_name)
    else:
        # Init the networks
        network.weights_init(generator, init_type = opt.init_type, init_gain = opt.init_gain)
        print('Initialize generator with %s type' % opt.init_type)
    return generator

def create_discriminator(opt):
    # Initialize the networks
    discriminator = network.PatchDiscriminator(opt)
    print('Discriminator is created!')
    # Init the networks
    network.weights_init(discriminator, init_type = opt.init_type, init_gain = opt.init_gain)
    print('Initialize discriminator with %s type' % opt.init_type)
    return discriminator
