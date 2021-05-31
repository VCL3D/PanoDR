import os
import cv2
import numpy as np
import tqdm
from vcl3datlantis.dataloaders.layout2sem import *
from vcl3datlantis.dataset.store_paths import loadPaths

root_path = r"D:\VCL\Dataset\Structured3D\Structured3D"

if __name__ == "__main__":
    if os.path.isdir(root_path):
        paths = [dirpath for dirpath, dirnames, _ in os.walk(root_path) if not dirnames]
    else:
        paths = loadPaths(root_path)  

    shape = None
    
    for path in tqdm.tqdm(paths):
        if shape is None:
            shape = cv2.imread(os.path.join(path, "semantic.png")).shape
        
        layout = np.loadtxt(os.path.join(os.path.dirname(path), "layout.txt"))

        layout_t, layout_viz = Layout(layout, shape)# 1 channel layout
        #get sem mask via top-bottom and bottom-up cumulative sum
        semantic_mask = Layout2Semantic(layout_t) # FCW

        labels_equ, semantic_mask_np = getLabels(semantic_mask)

        cv2.imwrite(os.path.join(path, "structure_semantics.png"), labels_equ.squeeze().float().numpy())
        


