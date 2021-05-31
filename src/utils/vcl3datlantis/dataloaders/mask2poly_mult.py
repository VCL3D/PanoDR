import numpy as np
#from imantics import Polygons, Mask
import PIL
from PIL import Image
#import shapely
#from shapely import geometry
#from shapely.geometry import MultiPolygon, Polygon
import random
from skimage import draw
import cv2
from matplotlib import path
import pycocotools.mask as mask_util
from PIL import Image, ImageDraw
import time
from random import randrange

def getScaledPoly(row, col, cx, cy, scale):
  col = np.array(col); row = np.array(row)
  row = list((  scale * (row - cx) ) + cx)
  col = list((  scale * (col - cy) ) + cy)
  return row, col

def get_conv_hull(msk, _type='gt'):
  #Input: semantic mask 
  #Output: convex hull mask
  contours, hierarchy = cv2.findContours(msk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  hull = []
  _mask = []
  # calculate points for each contour
  for i in range(len(contours)):
      # creating convex hull object for each contour
      hull.append(cv2.convexHull(contours[i], False))
  # create an empty image
  drawing = np.zeros((msk.shape[0], msk.shape[1], 3), np.uint8)

  color_contours = (0, 255, 0) # green - color for contours
  if _type == 'gt':
    color = (255, 255, 255) # blue - color for convex hull of gt mask
  elif _type == 'noisy':
    color = (255, 255, 255) # red - color for convex hull of noisy mask
  # draw contours and hull points
  for i in range(len(contours)):
      # draw ith contour & convex hull object
      cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
      _mask.append(cv2.drawContours(drawing, hull, i, color, 1, 8))
      #cv2.imshow(_type, drawing)
      #cv2.waitKey(0)    
  __mask = np.array(_mask).sum(axis=0)
  return __mask

def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask
"""
def getPolyVert(sem_mask):
  polygons = Mask(sem_mask[:,:,0]).polygons()
  row=[];col=[]

  for x in polygons.points:
      for j in range(len(x)):
          _row = x[j][0] 
          _col = x[j][1] 
          row.append(_row) #row vertices
          col.append(_col) #col vertices
  return row, col
"""
def findCentroid(indices, mean_val):
  return min(enumerate(indices), key=lambda x: abs(x[1]-mean_val))

def findMeanRowCol(row, col):
    mean_row = np.mean(np.array(row))
    mean_col = np.mean(np.array(col))
    return mean_row, mean_col
"""
def getScaledMask(sem_mask, scale_val):
    row, col = getPolyVert(sem_mask)
    #find centroid for polygon
    mean_row, mean_col = findMeanRowCol(row, col)
    idx, cx = findCentroid(row, mean_row)
    idy, cy = findCentroid(col, mean_col)
    #Upscale
    row_scaled, col_scaled = getScaledPoly(row, col, cx, cy, scale_val)
    row_scaled[idx]=cx;col_scaled[idy]=cy
    msk_large = poly2mask(col_scaled, row_scaled, (height, width))
    msk_large = msk_large.astype(np.uint8)
    return msk_large
"""
def getContour(_mask):
  _contour = []
  contours, hierarchy = cv2.findContours(_mask[:,:,0].astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  _contour.append(cv2.drawContours(_mask[:,:,0].astype(np.uint8), contours, -1, (255,255,255)))
  return contours, _contour

def getOneMask(contours, _contour, height, width):
    try:
      bigger = max(contours, key=lambda item: cv2.contourArea(item))
      the_mask = np.zeros((height,width),np.uint8)
      cv2.drawContours(the_mask, [bigger], -1, 255, thickness=cv2.FILLED)
      # cv2.imshow("the_mask", the_mask)
      # cv2.waitKey(0)
      f_hull = cv2.bitwise_and(_contour[0], _contour[0], mask = the_mask)
      cv2.fillPoly(f_hull, pts =[bigger], color=255)
      
    except ValueError:
      f_hull = None

    return f_hull.astype(np.bool)

if __name__ == '__main__':

  semantic_path = "D:/codes/PEN-Net-for-Inpainting/mask2poly/semantic.png"
  diminish_class = 6
  scaleUp = False
  sem_mask = np.array(Image.open(semantic_path), dtype=np.int32)
  sem_mask = (sem_mask==diminish_class).astype(int)
  height, width = sem_mask.shape

  sem_mask = np.repeat(sem_mask[:, :, np.newaxis], 3, axis=2)
  sem_mask = sem_mask.astype(np.uint8)
  scale_val = 1.15
  
  start = time.time()
  if scaleUp:
    msk_scaled = getScaledMask(sem_mask) 
    _mask = get_conv_hull(msk_scaled, 'noisy')
  else:
    _mask= get_conv_hull(sem_mask[:,:,0], 'gt')
  
  contours, _contour = getContour(_mask)
  #Pick up the biggest contour
  f_hull = getOneMask(contours, _mask, _contour)
  
  end = time.time()
  print(end - start)

  cv2.imwrite("D:/tmp/gt.png", sem_mask*255)
  cv2.imwrite("D:/tmp/_mask.png", _mask.astype(np.uint8))
  #cv2.imwrite("D:/tmp/_mask_large.png", _mask_large.astype(np.uint8))
  cv2.imwrite("D:/tmp/_mask_large_final.png", _contour[0].astype(np.uint8))
  cv2.imwrite("D:/tmp/_mask_large_final_removed.png", f_hull.astype(np.uint8))
  a=1
  #estimate time: about 0.027 seconds per image