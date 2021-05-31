from vcl3datlantis.models.semantic_segmentation_models.unet.model import UNet
from vcl3datlantis.dataloaders.Structured_3D_refined import DRS3D
from torch.utils.data import DataLoader
import os
import argparse
import torch
import tqdm
import cv2
import numpy as np

def prepareModel(args):
    model = UNet(3,3)
    model.load_state_dict(torch.load(args.pth))
    model.eval()
    return model.cuda()


def prepareDataset(args):
    data = DRS3D(args.data_path, args.width , args.height, 0.8, 0.01, True)
    dataloader = DataLoader(data, args.batch_size)
    return dataloader

def toOneHot(a):
    return (np.arange(a.max() + 1) == a[...,None]).astype(int)

def processData(args,model,dataloader):
    os.makedirs(args.output_dir, exist_ok = True)
    
    for b in tqdm.tqdm(dataloader):
        masked_img = torch.where(b["mask"].byte(), b["img"], torch.ones_like(b["img"])).cuda()
        y = model(masked_img)
        labels = torch.argmax(y,dim = 1, keepdim=True)

        for i in range(y.shape[0]):
            splited = b["img_path"][i].split("\\")
            name = splited[-4] + "_" + splited[-2] + "_semantic.png"
            labels_i = labels[i].cpu().squeeze().numpy()
            onehot = toOneHot(labels_i)
            cv2.imwrite(os.path.join(args.output_dir, name), onehot.astype(np.uint8) * 255)


def main(args):
    model = prepareModel(args)
    dataloader = prepareDataset(args)
    with torch.no_grad():
        processData(args, model, dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pth', required=True,
                        help='path to load saved checkpoint.')
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--width', default=512)
    parser.add_argument('--height', default=256)
    parser.add_argument('--batch_size', default=2)
    args = parser.parse_args()
    
    main(args)