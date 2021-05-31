import os
import cv2
import numpy as np
import random
import torch
import time
import glob
from PIL import Image
from functools import reduce
from torch.utils.data.dataset import Dataset
from vcl3datlantis.dataset.store_paths import loadPaths
from vcl3datlantis.dataloaders.misc.panorama import draw_boundary_from_cor_id
from vcl3datlantis.dataloaders.misc.utils import showImages, colorizeLabels
from vcl3datlantis.dataloaders.layout2sem import one_hot
from typing import Set,Tuple, Dict
from torch.utils.data import DataLoader
from skimage.feature import canny
from skimage.color import rgb2gray
import albumentations as A

FLOOR_ID = 2
WALL_ID = 1
CEILING_ID = 22
DEPTH_THRESHOLD = 100
# AUGMENT_OBJECT_TYPE = 24
UNMASKED_ID = -1
# INVALID_ID = -2
DONT_CARE_INFO = -1
HEIGHT_TOLERANCE_PERCENTAGE = 0.2 
VALID_OBJECT_SIZE_PERCENTAGE = 0.000762939453125 

def remap_segmentation(cwf : np.ndarray) -> np.ndarray:
    d = {2:0,1:2,3:1}
    cwf_copy = cwf.copy()
    for k, v in d.items(): cwf_copy[cwf==k] = v
    return cwf_copy

class DRS3D(Dataset):
    def __init__(self, 
                root_path : str, 
                width : int, 
                height : int, 
                max_mask_area : float = 1.0,
                min_mask_area : float = 0.001, #good value, dont change
                roll : bool = True, #rolling for augmentation
                seed : int = 1313,
                layout_extras : bool = False,
                object_mask_only : bool = True,
                load_edges      : bool = False,
                dilate_convex_mask : bool = True,
                return_full : bool = False,
                only_rawlight : bool = True,
                album: bool = False
                ) -> None:

        if isinstance(root_path, str):
            self._paths = glob.glob(f"{root_path}\\*\\*\\*\\*")
        elif isinstance(root_path, list):
            self._paths = root_path

        self._width, self._height = width, height
        self._S3D_WIDTH, self._S3D_HEIGHT = 1024, 512
        self._ratio_w , self._ratio_h = self._S3D_WIDTH / width, self._S3D_HEIGHT / height
        self._resize = (width != self._S3D_WIDTH) or (height != self._S3D_HEIGHT)
        if only_rawlight:
            self._room_lightings = ["rawlight"]
        else:
            self._room_lightings = ["rawlight", "coldlight", "warmlight"]
        self._min_mask_area, self._max_mask_area = min_mask_area, max_mask_area
        self._classes4masking = {3, 4, 5, 6, 7, 10, 11, 12, 14, 15, 17, 19, 24, 25, 29, 30, 32, 33, 34, 36} #40
        self._roll = roll
        self._layout_extras = layout_extras
        self._random_mask_side_size_percentage = 0.3
        self._random_mask_side_deviation_percentage = 0.1
        self._object_mask_only = object_mask_only
        self._load_edges = load_edges
        self._dilate_convex_mask = dilate_convex_mask
        self._return_full = return_full
        self.seed = seed
        self.rng = random.Random(seed)
        self.augmented = None
        self.transformation = A.RandomBrightnessContrast(p=0.5)
        self.album = album
        
    
    def _read_data(self, path : str, type : str):
        
        empty_panorama = cv2.imread(os.path.join(path, "empty", f"rgb_{type}.png"))
        empty_panorama = cv2.cvtColor(empty_panorama, cv2.COLOR_BGR2RGB)
        empty_semantic_map = np.array(Image.open(os.path.join(path,"empty", "semantic.png")), dtype=np.int32)
        
        full_panorama = cv2.imread(os.path.join(path, "full", f"rgb_{type}.png"))
        full_panorama = cv2.cvtColor(full_panorama, cv2.COLOR_BGR2RGB)

        if self.album:
            self.augmented = self.transformation(image=empty_panorama, image0=full_panorama)
            empty_panorama = self.augmented['image']
            full_panorama = self.augmented['image0']

        full_semantic_map = np.array(Image.open(os.path.join(path,"full", "semantic.png")), dtype=np.int32)

        cwf = cv2.imread(os.path.join(path,"empty", "structure_semantics.png"), 0)
        cwf = remap_segmentation(cwf)

        if self._layout_extras:
            normals = cv2.imread(os.path.join(path,"empty", "normal.png"), -1)
            depth = cv2.imread(os.path.join(path,"empty", "depth.png"), -1)
        else:
            normals = None
            depth = None

        if self._resize:
            target_size = (self._width, self._height)
            empty_panorama = cv2.resize(empty_panorama, target_size, cv2.INTER_CUBIC)
            empty_semantic_map = cv2.resize(empty_semantic_map.astype(np.uint8), target_size, cv2.INTER_NEAREST)

            full_panorama = cv2.resize(full_panorama, target_size, cv2.INTER_CUBIC)
            full_semantic_map = cv2.resize(full_semantic_map.astype(np.uint8), target_size, cv2.INTER_NEAREST)

            cwf = cv2.resize(cwf.astype(np.uint8), target_size, cv2.INTER_NEAREST)

            if self._layout_extras:
                normals = cv2.resize(normals, target_size, cv2.INTER_NEAREST)
                depth = cv2.resize(depth, target_size, cv2.INTER_NEAREST)

        if self._roll:
            self.roll_size = self.rng.randint(0, self._width - 1)
            cwf = np.roll(cwf, shift = self.roll_size, axis = 1)
            empty_panorama = np.roll(empty_panorama, shift = self.roll_size, axis = 1)
            empty_semantic_map = np.roll(empty_semantic_map, shift = self.roll_size, axis = 1)
            full_panorama = np.roll(full_panorama, shift = self.roll_size, axis = 1)
            full_semantic_map = np.roll(full_semantic_map, shift = self.roll_size, axis = 1)

            if self._layout_extras:
                normals = np.roll(normals, shift = self.roll_size, axis = 1)
                depth = np.roll(depth, shift = self.roll_size, axis = 1)
                


        return empty_panorama, empty_semantic_map,\
                full_panorama, full_semantic_map, cwf, \
                normals, depth

    def _extract_scene_objects(self, empty_semantic_map : np.ndarray, full_semantic_map : np.ndarray) -> Tuple[Set,Set,Set]:
        
        objects_in_empty = np.unique(empty_semantic_map.flatten())
        objects_in_full = np.unique(full_semantic_map.flatten())
        return set(objects_in_empty), set(objects_in_full)
    
    def _select_candidates(self, objects_in_empty : Set[int],
                                 objects_in_full : Set[int]) -> Tuple[int]:
        return (objects_in_full - objects_in_empty).intersection(self._classes4masking)
        #return (objects_in_full).intersection(self._classes4masking)

    def _produce_random_mask(self) -> np.ndarray:
        min_size = min(self._width, self._height)
        width_size = int(min_size * (self._random_mask_side_size_percentage + self.rng.random() * self._random_mask_side_deviation_percentage))
        height_size = int(min_size * (self._random_mask_side_size_percentage + self.rng.random() * self._random_mask_side_deviation_percentage))

        pos_x = self.rng.randint(0, self._width - width_size - 1)
        pos_y = self.rng.randint(0, self._height - height_size - 1)

        mask = np.zeros((self._height, self._width), dtype = np.uint8)
        mask[pos_y : pos_y + height_size, pos_x : pos_x + width_size] = 255
        return mask



    def _compute_mask(self, semantic_map : np.ndarray, candidate_objects_for_removal : list) -> np.ndarray:
        image_area = semantic_map.size
        while len(candidate_objects_for_removal):
            chosen_id = self.rng.choice(candidate_objects_for_removal)
            object_mask = (semantic_map == chosen_id).astype(np.uint8) * 255

            # https://stackoverflow.com/questions/25504964/opencv-python-valueerror-too-many-values-to-unpack
            contours, _ = cv2.findContours(object_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            # Find the convex hull object for each contour

            boundary_objects = []
            for i in range(len(contours)):
                for p in contours[i]:
                    if (0 in p) or ((self._width - 1) in p):
                        boundary_objects.append(i)
            
            if len(boundary_objects) > 1:
                h = np.zeros_like(object_mask)
                for i in range(len(boundary_objects)):
                    hull = cv2.convexHull(contours[boundary_objects[i]])
                    h = cv2.fillConvexPoly(h, hull, (1))
                area = np.sum(h)
                if (area > self._min_mask_area * image_area) and (area < self._max_mask_area * image_area):
                    return h * 255
                else:
                    candidate_objects_for_removal.remove(chosen_id)
                    continue
            elif len(contours) >= 1:
                max_area , max_id = -1,-1
                for i in range(len(contours)):
                    hull = cv2.convexHull(contours[i])
                    h = np.zeros_like(object_mask)
                    h = cv2.fillConvexPoly(h, hull, (1))
                    area = np.sum(h)
                    if (area > max_area) and (area > self._min_mask_area * image_area) and (area < self._max_mask_area * image_area):
                        max_area, max_id = area, i
                        mask = h
                if max_id != -1:
                    return mask * 255
                else: #suitable object not found
                    candidate_objects_for_removal.remove(chosen_id)
                    continue
            return None
        return None

    def _load_image_edges(self, image : np.ndarray, sigma : float) -> np.ndarray:
        
        edges = canny(image, sigma=sigma).astype(np.float)
        return edges

    def fetch(self, i):
        path = self._paths[i]
        light_type = self.rng.choice(self._room_lightings)

        try:
            empty_rgb, empty_semantic, full_rgb, full_semantic, cwf, normals, depth = self._read_data(path, light_type)
        except Exception as e:
            print(f"error reading {path}")
            return None

        foreground = 255 * ((full_semantic != CEILING_ID).astype(np.uint8) * (full_semantic != FLOOR_ID).astype(np.uint8) * (full_semantic != WALL_ID).astype(np.uint8))
        augmented = np.where(foreground.astype(np.bool)[...,None], full_rgb, empty_rgb)

        objects_empty, objects_full = self._extract_scene_objects(empty_semantic, full_semantic)
        candidate_objects_for_removal = self._select_candidates(objects_empty, objects_full)

        if len(candidate_objects_for_removal) == 0:
            if self._object_mask_only:
                return None
            else:
                mask = self._produce_random_mask()
        else:
            mask = self._compute_mask(full_semantic, list(candidate_objects_for_removal))
            if mask is None:
                if self._object_mask_only:
                    return None
                else:
                    mask = self._produce_random_mask()
            else:
                if self._dilate_convex_mask:
                    kernel = np.ones((3,3), np.uint8)
                    mask = cv2.dilate(mask, kernel, iterations = 1, borderValue = 255)
                  

        cwf_t = torch.from_numpy(cwf).float()
        cwf_one_hot = one_hot(cwf_t.unsqueeze(0), 3)

        ret_dict = {
            "img" : torch.from_numpy(augmented).permute(2,0,1).float()/255.0,
            "mask" : 1 - (torch.from_numpy(mask).unsqueeze(0).float() / 255.0),
            "img_gt" : torch.from_numpy(empty_rgb).permute(2,0,1).float()/255.0,
            "img_path" : path,
            "label_semantic" : cwf_t.unsqueeze(0),
            "label_one_hot" : cwf_one_hot
        }

        if self._load_edges:
            sigma = 0
            blur_k = (5,5)
            img_gray_empty = rgb2gray(empty_rgb)
            img_gray_full = rgb2gray(full_rgb)

            edges_filepath = os.path.join(path, "empty", "edges.png")
            if not os.path.exists(edges_filepath) or True:
                edges_empty = self._load_image_edges(cv2.blur(img_gray_empty, blur_k), sigma)
                #cv2.imwrite(edges_filepath, edges_empty * 255)
                #cv2.imshow("edges_empty", edges_empty.astype(np.uint8) * 255)
            else:
                edges_empty = cv2.imread(edges_filepath, 0).astype(np.float64) / 255

            edges_filepath = os.path.join(path, "full", "edges.png")
            if not os.path.exists(edges_filepath) or True:
                edges_full = self._load_image_edges(cv2.blur(img_gray_full, blur_k), sigma)
                #cv2.imwrite(edges_filepath, edges_full * 255)
            else:
                edges_full = cv2.imread(edges_filepath, 0).astype(np.float64) / 255
                #cv2.imshow("edges_full", edges_full.astype(np.uint8) * 255)


            edges = np.where(mask.astype(np.bool),edges_empty, edges_full)
            # cv2.imshow("edges", edges.astype(np.uint8) * 255)
            img_gray = np.where(mask.astype(np.bool),img_gray_empty, img_gray_full)
            
            # cv2.imshow("img_gray", img_gray)
            # cv2.waitKey(0)
            ret_dict.update({"edges" : torch.from_numpy(edges).float().unsqueeze(0) })
            ret_dict.update({"img_gray" : torch.from_numpy(img_gray).float().unsqueeze(0) })

        if self._return_full:
            ret_dict.update({"full": torch.from_numpy(full_rgb).permute(2,0,1).float()/255.0})


        return ret_dict


    def __getitem__(self, i):
        item = self.fetch(i)
        while item is None:
            #self._paths.remove(self._paths[i])
            new_i = self.rng.randint(0, self.__len__() - 1)
            item = self.fetch(new_i)
        return item


    def __len__(self):
        return len(self._paths) 


if __name__ == "__main__":
    dataset = DRS3D("Path",1024,512, 0.8, 0.01,  roll = True, layout_extras = False, object_mask_only=True, load_edges=True, dilate_convex_mask=True)
    dataloader = DataLoader(dataset, batch_size = 1, shuffle=True)
    counter = 0
    import tqdm
    timer = 0
    torch.manual_seed(0)
    for b in dataloader:
        img = b["img"]
        mask = b["mask"]
        img_gt = b["img_gt"]
        masked = torch.where(mask.bool(), img, torch.ones_like(img))
        print(b["img_path"])
        showImages(
            masked = masked[0].permute(1,2,0).numpy(),
            gt = img_gt[0].permute(1,2,0).numpy(),
            img = img[0].permute(1,2,0).numpy(),
            mask = mask[0].permute(1,2,0).numpy(),
        )