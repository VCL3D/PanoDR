import tqdm
import os
import argparse
import cv2
import torch
import numpy
from torch.utils.data import DataLoader
from vcl3datlantis.dataloaders.Structured3D import DatasetStructure3D
from pytorch_lightning.trainer import seed_everything

def str2bool(x : str):
    return x.lower() == "true"

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', required=True)
parser.add_argument('--out_path', required=True)
parser.add_argument('--target_size', type=int, default=256)
parser.add_argument('--one_folder', type = str2bool, default = "false")
args = parser.parse_args()


if __name__ == "__main__":
    seed_everything(12345) # do not change
    os.makedirs(args.out_path, exist_ok=True)

    dataloader = DataLoader(DatasetStructure3D(
            root_path = args.in_path,
            width = args.target_size, height = args.target_size,
            layout="full", datum_type="normal", color = "rawlight, warmlight, coldlight",
            mask_reverse = True, normalize = None,
            object_size_percentage = (0.1,0.4), config_path = None,
            ), 
            batch_size = 1, 
            shuffle = False, 
            num_workers = 0)

    if args.one_folder:
        for item in tqdm.tqdm(dataloader):
            for i in range(len(item["img_path"])):
                splited_path = item["img_path"][i].split("\\")
                scene, room, lighttype = splited_path[-6],splited_path[-4],splited_path[-1].split('.')[0]
                mask = (item["mask"][i].squeeze().cpu()).byte()
                img = (item["img"][i].cpu() * 255).byte()
                img_gt = (item["img_gt"][i].cpu() * 255).byte()
                img_to_save = torch.where(mask.byte(), img,img_gt)
                out_path = os.path.join(args.out_path,scene + "_room_" + room + "_" + lighttype + ".png")
                cv2.imwrite(out_path, img_to_save.permute(1,2,0).cpu().numpy())


    else:
        for item in tqdm.tqdm(dataloader):
            for i in range(len(item["img_path"])):
                splited_path = item["img_path"][i].split("\\")
                scene, room, lighttype = splited_path[-6],splited_path[-4],splited_path[-1].split('.')[0]
                mask = (item["mask"][i].squeeze().cpu() * 255).byte().numpy()
                bbox = item["bbox"][i].squeeze().numpy()
                img = (item["img"][i].cpu().permute(1,2,0) * 255).byte().numpy()
                img_gt = (item["img_gt"][i].cpu().permute(1,2,0) * 255).byte().numpy()
                label_semantic = item["label_semantic"][i].cpu().squeeze().numpy()
                label_one_hot = (item["label_one_hot"][i].cpu().permute(1,2,0).squeeze() * 255).byte().numpy()

                out_path = os.path.join(args.out_path,scene + "_room_" + room + "_" + lighttype)
                os.makedirs(out_path, exist_ok=True)
                

                cv2.imwrite(os.path.join(out_path, "img.png"), img)
                cv2.imwrite(os.path.join(out_path, "img_gt.png"), img_gt)
                cv2.imwrite(os.path.join(out_path, "mask.png"), mask)
                cv2.imwrite(os.path.join(out_path, "label_semantic.png"), label_semantic)
                cv2.imwrite(os.path.join(out_path, "label_one_hot.png"), label_one_hot)
                numpy.savetxt(os.path.join(out_path, "bbox.txt"), bbox, fmt = '%d', header = "xmin,ymin,xmax,ymax")
