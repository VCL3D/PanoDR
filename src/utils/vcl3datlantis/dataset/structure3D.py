import torch
import os
import cv2
import sys
import numpy as np
from numpy import random as np_random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import data_utils
from enum import Enum

from torch.utils.data.dataset import Dataset
from numpy.random import default_rng

class TargetSelectionStrategy(Enum):
    RANDOM = 1,
    BOTTOM_CORNERS = 2

class DatasetStructure3D(Dataset):
    def getItemFunctionFactory(
        self,
        strategy : TargetSelectionStrategy
    ):
        if strategy is TargetSelectionStrategy.RANDOM:
            if not hasattr(self, "angle_max"):
                raise RuntimeError(strategy.name + " requires angle_max parameter in DatasetStructure3D initialization")
            else:
                def g(i):
                    phi, theta = self.rng.random(), 2 * (self.rng.random() - 0.5) * self.angle_max + 0.5 #phi, theta
                    filename = os.path.join(self._paths[i//len(self.lighting)], self.map_lighting_to_filename[i % len(self.lighting)])
                    tensor = torch.from_numpy(cv2.imread(filename))
                    out = self.nforv_operation.toNFOV(tensor, np.array([phi, theta])).permute(2,1,0)
                    return {
                        "filename" : filename,
                        "image" : out
                        }
                return g

        elif strategy is TargetSelectionStrategy.BOTTOM_CORNERS:
            def g(i):
                filename = os.path.join(self._paths[i//len(self.lighting)], self.map_lighting_to_filename[i % len(self.lighting)])
                tensor = torch.from_numpy(cv2.imread(filename))
                layout_file = os.path.join(os.path.dirname(self._paths[i//len(self.lighting)]), "layout.txt")
                layout = np.loadtxt(layout_file)
                ### ATTENTION THIS MIGHT BE BUGGY
                ### ASSUMPTIONS FOR THE PIECE OF CODE BELOW
                ### i) layout txt is formated as width, height
                ### ii) data come in pairs that make up two walls intersections
                ### iii) first is always bottom corner, second is always top corner
                number_of_wall_intersections = layout.shape[0] / 2
                target_corner = int(self.rng.random() * number_of_wall_intersections)
                phi , theta = layout[target_corner * 2 + 1, 0] / tensor.shape[1] , layout[target_corner * 2 + 1, 1] / tensor.shape[0]
                out = self.nforv_operation.toNFOV(tensor, np.array([phi, theta])).permute(2,1,0)
                return {
                        "filename" : filename,
                        "image" : out
                        }
            return g
        else:
            raise RuntimeError(strategy.name + " is not a valid TargetSelectionStrategy")

    def __init__(   self,
                    root_path,
                    width,
                    height,
                    target_selection_strategy = TargetSelectionStrategy.BOTTOM_CORNERS,
                    seed = 271092,
                    **kwargs
        ):
        '''
        root_path   :   str     Path to root of structure3D dataset (where scenes are located)
        empty       :   bool    Whether load empty layout
        simple      :   bool    Whether load simple layout
        full        :   bool    Whether load full layout
        albedo      :   bool    Whether load albedo
        depth       :   bool    Whether load depth
        normal      :   bool    Whether load normal
        coldlight   :   bool    Whether load coldlight
        rawlight    :   bool    Whether load rawlight
        warmlight   :   bool    Whether load warmlight
        semantic    :   bool    Whether load semantic
        '''


        super().__init__()

        for arg in kwargs:
            setattr(self, arg, kwargs[arg])
            


        self.rng = default_rng(seed = seed)
        self.map_type_to_filename = {
            "albedo"            : "albedo.png",
            "depth"             : "depth.png",
            "normal"            : "normal.png",
            "semantic"          : "semantic.png"
        }

        # self.map_lighting_to_filename = {
        #     "rawlight"          : "rgb_rawlight.png",
        #     "warmlight"         : "rgb_warmlight.png",
        #     "coldlight"         : "rgb_coldlight.png"
        # }


        self.map_lighting_to_filename = [
            "rgb_rawlight.png",
            "rgb_warmlight.png",
            "rgb_coldlight.png"
        ]

        self.layouts = []
        self.dataTypes = []

        for layout in ["empty", "simple", "full"]:
            for key, value in kwargs.items():
                if isinstance(value, str):
                    if layout in value:
                        self.layouts.append(layout)

        assert len(self.layouts), "Please choose layout type [empty, simple, full]"

        for datum_type in ["albedo", "depth", "normal", "semantic"]:
            for key, value in kwargs.items():
                if isinstance(value, str):
                    if datum_type in value:
                        self.dataTypes.append(datum_type)

        assert len(self.dataTypes), "Please choose layout type [albedo, depth, normal, semantic]"

        self.lighting = []
        for light in ["rawlight", "warmlight", "coldlight"]:
            for key, value in kwargs.items():
                if isinstance(value, str):
                    if light in value:
                        self.lighting.append(light)
        
        assert len(self.lighting), "Please choose lighting type [rawlight, warmlight, coldlight]"

        #self._paths = [os.path.dirname(dirpath) for dirpath, dirnames, _ in os.walk(root_path) if not dirnames]
        self._paths = [dirpath for dirpath, dirnames, _ in os.walk(root_path) if not dirnames]

        self.nforv_operation = data_utils.camera.NFOVTORCH(height, width)

        self.get_item = self.getItemFunctionFactory(strategy=target_selection_strategy)
        
    

    def __getitem__(self, i):
        return self.get_item(i)

    def __len__(self):
        return len(self._paths) * len(self.lighting)

if __name__ == "__main__":
    path = r"Path"
    data_iter = DatasetStructure3D(root_path = path, width=640, height=480, angle_max = 0.3, layout="empty", datum_type="semantic", color = "rawlight, warmlight, coldlight")

    dataset =  torch.utils.data.DataLoader(data_iter,\
        batch_size = 2, shuffle=True,\
        num_workers = 0, pin_memory=False)

    for batch_id, batch in enumerate(dataset):
        for i in range(len(batch)):
            cv2.imshow(batch["filename"][i], batch["image"][i].permute(2,1,0).cpu().numpy())
            cv2.waitKey(0)






