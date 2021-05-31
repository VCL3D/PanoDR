from vcl3datlantis.models.PanoDR.layer import init_weights
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from vcl3datlantis.models.GatedConv.network_module import *
from vcl3datlantis.models.PanoDR.SEAN.normalization import * 
from vcl3datlantis.models.PanoDR.SEAN.spade_arch import *
from vcl3datlantis.models.semantic_segmentation_models.unet.model import UNet

class GatedGenerator(nn.Module):
    def __init__(self, opt, device):
        super(GatedGenerator, self).__init__()
        self.opt = opt
        if self.opt.use_argmax:
            self._sem_layout = None
            self.sem_layout = torch.FloatTensor(self.opt.batch_size, self.opt.in_layout_channels, self.opt.height, self.opt.width).to(device)
            self.sem_layout.zero_()
        if opt.structure_model == "unet":
           self.structure_model = UNet(n_channels = opt.in_layout_channels, n_classes = opt.num_classes)

        if opt.type_sp == 'SEAN':
            self.Zencoder = Zencoder(opt.in_SEAN_channels, opt.style_code_dim)
            self.spade_block_1 = SPADEResnetBlock(opt.latent_channels*4, opt.latent_channels*4, opt.in_spade_channels, device=device, Block_Name='up_0')
            self.spade_block_2 = SPADEResnetBlock(opt.latent_channels*2, opt.latent_channels*2, opt.in_spade_channels, device=device, Block_Name='up_1')
        
        self.refinement = nn.Sequential(
            # Surrounding Context Encoder
            GatedConv2d(opt.in_channels, opt.latent_channels, 5, 2, 2, pad_type = opt.pad_type, activation = opt.activation, norm='none'),
            GatedConv2d(opt.latent_channels, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 4, 3, 2, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 2, dilation = 2, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 4, dilation = 4, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 8, dilation = 8, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 16, dilation = 16, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation, norm = opt.norm),
            GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 4, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation),
        )
        #Structure-Aware Decoder
        self.refine_dec_1 = nn.Sequential(nn.Upsample(scale_factor=2),
        GatedConv2d(opt.latent_channels * 4, opt.latent_channels * 2, 3, 1, 1, activation = opt.activation, pad_type='zero', norm=opt.norm),
        )
        self.refine_dec_2 =  GatedConv2d(opt.latent_channels * 2, opt.latent_channels * 2, 3, 1, 1, pad_type = opt.pad_type, activation = opt.activation)
        self.refine_dec_3 = nn.Sequential(nn.Upsample(scale_factor=2), 

        GatedConv2d(opt.latent_channels * 2, opt.latent_channels, 3, 1, 1, pad_type ='zero', activation = opt.activation),
        )
        self.refine_dec_4 = GatedConv2d(opt.latent_channels, opt.out_channels, 3, 1, 1, pad_type = opt.pad_type, norm='none', activation = 'tanh')


    def forward(self, img, inverse_mask, masked_input, device, use_sean):

        first_out = masked_input
        structure_model_output = self.structure_model(masked_input).clone() 

        if self.opt.use_argmax:
            self._sem_layout = torch.softmax(structure_model_output, dim = 1)
            self._sem_layout = torch.argmax(self._sem_layout, dim=1, keepdim=True)
            self._sem_layout = torch.clamp(self._sem_layout, min=0, max=2)
            self.sem_layout.scatter_(1, self._sem_layout, 1)

        else:
            self.sem_layout = torch.softmax(structure_model_output, dim = 1)

        second_out = self.refinement(torch.cat((masked_input, inverse_mask), 1))  #Enconder + bottleneck

        if self.opt.type_sp == 'SEAN':
            style_codes = self.Zencoder(input=img, segmap=self.sem_layout)
            z = self.spade_block_1(second_out, self.sem_layout, style_codes)
            second_out = self.refine_dec_1(z)
            second_out = self.refine_dec_2(second_out)
            z = self.spade_block_2(second_out, self.sem_layout, style_codes)
            second_out = self.refine_dec_3(z)
            second_out = self.refine_dec_4(second_out)
        else: 
            second_out = self.refine_dec_1(second_out)
            second_out = self.refine_dec_2(second_out)
            second_out = self.refine_dec_3(second_out)
            second_out = self.refine_dec_4(second_out)
        second_out = torch.clamp(second_out, 0, 1)

        return first_out, second_out, structure_model_output, self.sem_layout

class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False, getIntermFeatures=True):
        super(Discriminator, self).__init__()
        self.n_layers = n_layers
        self.getIntermFeatures = getIntermFeatures

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),True),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),True),

                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                                    kernel_size=kw, stride=2, padding=padw, bias=use_bias), True),

            nn.LeakyReLU(0.2, True)
        ]

        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),True)]

        if getIntermFeatures:
            for n in range(len(sequence)):
                setattr(self, 'Dmodel'+str(n), nn.Sequential(sequence[n]))

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence) 

    def forward(self, img, mask, getIntermFeatures):
        self.getIntermFeatures = getIntermFeatures
        input = torch.cat((img,mask), 1)

        if self.getIntermFeatures:
            _out = [input]
            for n in range(self.n_layers+6):
                model = getattr(self, 'Dmodel'+str(n)) #Dmodel
                _out.append(model(_out[-1]))
            #return _out[1:]
            return _out[9], _out[1:6]
        else:
            out = self.model(input)
            return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
