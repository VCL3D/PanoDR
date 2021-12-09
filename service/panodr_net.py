import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import logging
from panodr_module import *
logger = logging.getLogger(__name__)

class GatedGenerator(nn.Module):
    def __init__(self):
        super(GatedGenerator, self).__init__()
        self.latent_channels = 64
        self.in_spade_channels = 3
        self.in_SEAN_channels = 3
        self.pad_type = 'spherical'
        self.activation = 'relu'
        self.norm = 'in'
        self.out_channels = 3
        self.use_Unet = False

        self.structure_model = UNet(n_channels = 3, n_classes = 3)
        self.Zencoder = Zencoder(self.in_SEAN_channels, 512)
        self.spade_block_1 = SPADEResnetBlock(self.latent_channels*4, self.latent_channels*4, self.in_spade_channels,  Block_Name='up_0')
        self.spade_block_2 = SPADEResnetBlock(self.latent_channels*2, self.latent_channels*2, self.in_spade_channels,  Block_Name='up_1')

        self.refinement = nn.Sequential(

            GatedConv2d(4, self.latent_channels, 5, 2, 2, pad_type = self.pad_type, activation = self.activation, norm='none'),
            GatedConv2d(self.latent_channels, self.latent_channels * 2, 3, 1, 1, pad_type = self.pad_type, activation = self.activation, norm = self.norm),
            GatedConv2d(self.latent_channels * 2, self.latent_channels * 4, 3, 2, 1, pad_type = self.pad_type, activation = self.activation, norm = self.norm),
            GatedConv2d(self.latent_channels * 4, self.latent_channels * 4, 3, 1, 1, pad_type = self.pad_type, activation = self.activation, norm = self.norm),
            # Bottleneck
            GatedConv2d(self.latent_channels * 4, self.latent_channels * 4, 3, 1, 1, pad_type = self.pad_type, activation = self.activation, norm = self.norm),
            GatedConv2d(self.latent_channels * 4, self.latent_channels * 4, 3, 1, 1, pad_type = self.pad_type, activation = self.activation, norm = self.norm),
            GatedConv2d(self.latent_channels * 4, self.latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = self.pad_type, activation = self.activation, norm = self.norm),
            GatedConv2d(self.latent_channels * 4, self.latent_channels * 4, 3, 1, 4, dilation = 4, pad_type = self.pad_type, activation = self.activation, norm = self.norm),
            GatedConv2d(self.latent_channels * 4, self.latent_channels * 4, 3, 1, 8, dilation = 8, pad_type = self.pad_type, activation = self.activation, norm = self.norm),
            GatedConv2d(self.latent_channels * 4, self.latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = self.pad_type, activation = self.activation, norm = self.norm),
            GatedConv2d(self.latent_channels * 4, self.latent_channels * 4, 3, 1, 1, pad_type = self.pad_type, activation = self.activation, norm = self.norm),
            GatedConv2d(self.latent_channels * 4, self.latent_channels * 4, 3, 1, 1, pad_type = self.pad_type, activation = self.activation),
        )
        self.refine_dec_1 = nn.Sequential(nn.Upsample(scale_factor=2),
        GatedConv2d(self.latent_channels * 4, self.latent_channels * 2, 3, 1, 1, activation = self.activation, pad_type='zero', norm=self.norm),
        )
        self.refine_dec_2 =  GatedConv2d(self.latent_channels * 2, self.latent_channels * 2, 3, 1, 1, pad_type = self.pad_type, activation = self.activation)
        self.refine_dec_3 = nn.Sequential(nn.Upsample(scale_factor=2), 

        GatedConv2d(self.latent_channels * 2, self.latent_channels, 3, 1, 1, pad_type ='zero', activation = self.activation),
        )
        self.refine_dec_4 = GatedConv2d(self.latent_channels, self.out_channels, 3, 1, 1, pad_type = self.pad_type, norm='none', activation = 'tanh')

    def forward(self, data):
        image = data[0,:,:,:].unsqueeze_(0)
        inverse_mask = (data[1,:,:,:].unsqueeze_(0))[:,0,:,:].unsqueeze_(0)
        masked_input = image * (1.0-inverse_mask) + inverse_mask

        #_sem_layout = None
        _sem_layout = torch.zeros_like(image)
        sem_layout = torch.zeros_like(image)

        structure_model_output = self.structure_model(masked_input).clone() 
        if self.use_Unet:
            #old
            sem_layout = None
            sem_layout = torch.softmax(structure_model_output, dim = 1)
            _sem_layout = torch.argmax(sem_layout, dim=1, keepdim=True)
            _sem_layout = torch.clamp(_sem_layout, min=0, max=2)
            sem_layout.scatter_(1, _sem_layout, 1)

        else:
            sem_layout = data[2,:,:,:].unsqueeze_(0) 
            sem_layout = sem_layout[0,:,:]
            sem_layout = torch.softmax(structure_model_output, dim = 1)
            _sem_layout = torch.argmax(sem_layout, dim=1, keepdim=True)
            _sem_layout = torch.clamp(_sem_layout, min=0, max=2)
            sem_layout.scatter_(1, _sem_layout, 1)

        second_out = self.refinement(torch.cat((masked_input, inverse_mask), 1))  #Enconder + bottleneck
        style_codes = self.Zencoder(input=image, segmap=sem_layout)
        z = self.spade_block_1(second_out, sem_layout, style_codes)
        second_out = self.refine_dec_1(z)
        second_out = self.refine_dec_2(second_out)
        z = self.spade_block_2(second_out, sem_layout, style_codes)
        second_out = self.refine_dec_3(z)
        second_out = self.refine_dec_4(second_out)
        second_out = torch.clamp(second_out, 0, 1)

        #get the diminished img
        output = image * (1.0-inverse_mask) + second_out * inverse_mask

        return output, sem_layout, second_out

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module