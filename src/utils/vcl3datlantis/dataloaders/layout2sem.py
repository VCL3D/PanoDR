import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from .mask2poly_mult import *
from .misc.panorama import draw_boundary_from_cor_id
from .misc.colors import colormap_255 
import torch
import torch.nn as nn

def Layout(layout,img_shape):
    layout_viz, _, _ = draw_boundary_from_cor_id(layout, img_shape)        
    layout_t = torch.from_numpy(layout_viz)
    return layout_t, layout_viz

def Layout2Semantic(layout_t):
    top_bottom = layout_t.cumsum(dim=0) > 0
    bottom_up = 2*(torch.flipud(torch.flipud(layout_t).cumsum(dim=0) > 0))
    semantic_mask = top_bottom + bottom_up
    return semantic_mask

def VizWithClasses(mask_np):
    h,w = mask_np.shape[:2]
    viz = np.zeros((h,w,3))
    
    viz[mask_np==0,:] = (255,0,0)      #ceiling: red
    viz[mask_np==1,:] = (0,0,255)      #floor: blue
    viz[mask_np==2,:] = (255,255,255)  #wall: white
    return viz

def getLabels(semantic_mask):
    semantic_mask_np = semantic_mask.detach().cpu().numpy() 
    labels = semantic_mask.unsqueeze(0).long()
    return labels, semantic_mask_np

def one_hot(labels, C):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    
    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size. 
        Each value is an integer representing correct classification.
    C : integer. 
        number of classes in labels.  
    
    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor   
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    #one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
    one_hot = torch.FloatTensor(C, labels.size(1), labels.size(2)).zero_()
    target = one_hot.scatter_(0, labels.long(), 1)  
    #target = Variable(target)
        
    return target

def getRoomPath(scene_path, room_id):
    room_path = os.path.join(scene_path, room_id, "panorama")
    return room_path

def convert_3_classes(labels_persp):
    _labels_persp = torch.zeros(labels_persp.shape)
    _labels_persp[labels_persp==1]=0;_labels_persp[labels_persp==2]=1;_labels_persp[labels_persp==3]=2
    return _labels_persp

