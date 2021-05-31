
import torch
import torch.nn.functional as F
import numpy as np
import random
import cv2
from itertools import product, permutations
import time

def getStructureImages(gt_label_one_hot):
    image_floor = gt_label_one_hot[:,0,:,:].unsqueeze(1).float()
    image_ceiling = gt_label_one_hot[:,1,:,:].unsqueeze(1).float()
    image_wall = gt_label_one_hot[:,2,:,:].unsqueeze(1).float()
    return image_floor, image_ceiling, image_wall    

def splitImageToStructures(images, image_floor, image_ceiling, image_wall):
    mult_floor = (images*image_floor).unsqueeze(2)
    mult_ceiling = (images*image_ceiling).unsqueeze(2)
    mult_wall = (images*image_wall).unsqueeze(2)
    return mult_floor, mult_ceiling, mult_wall

def getClassMapping():
    Layout_map = np.array([
    [255.        , 0.        , 0.        ],
    [0.        , 0.        , 255.        ],
    [255, 255, 255]])
    return(Layout_map)

def unNormalizeCoords(ix,iy, h, w):
    ix = ((ix + 1) / 2) * (w-1)
    iy = ((iy + 1) / 2) * (h-1)
    return ix.int(), iy.int()

def drawArrows(mask_bbox, comp, regression_input, batch_size, image_size):
    #fix #5
    b, c, h, w = comp.shape
    arrowsViz = torch.ones((batch_size, c, h, w))
    isValid = True

    regression_input_d = torch.clamp(regression_input.detach(),min  = -1, max = 1)
    
    for b in range(batch_size):
        _bbox = mask_bbox[b].cpu().numpy()
        x_min, y_min, x_max, y_max = _bbox
        #y_min, x_min, y_max, x_max = _bbox
        dx,dy = w // regression_input_d.shape[3], h // regression_input_d.shape[2]
        coordsX, coordsY = unNormalizeCoords(regression_input_d[b, 0, :, :], regression_input_d[b, 1, :, :], h, w)
        comp_np = np.float32(comp[b,:, :, :].permute(1,2,0).detach().cpu().numpy()).copy()

        unif_x = int((x_max-x_min)*0.20)
        unif_y = int((y_max-y_min)*0.20)
        try:
            list_coordsX = list(range(x_min, x_max, unif_x))
            list_coordsY = list(range(y_min, y_max, unif_y))
            #points = list(zip(list_coordsX, list_coordsY))
            points = list(product(list_coordsX, list_coordsY))

            for j in range(len(points)):
                x = points[j][0]
                y = points[j][1]
                cv2.arrowedLine(comp_np, (x, y), (coordsX[y//dy,x//dx], coordsY[y//dy,x//dx]), (1,0,1), 1)
            arrowsViz[b,:, :, :] = torch.from_numpy(comp_np).permute(2,0,1)

        except ValueError:
            isValid=False

    return arrowsViz, isValid