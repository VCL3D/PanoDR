import os
import pathlib
import torch
import numpy as np
import imageio
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from vcl3datlantis.metrics.inception import InceptionV3
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
import glob
import argparse
from lpips_pytorch import LPIPS, lpips

class Reconstruction_Metrics():
    def __init__(self, device, metric_list=['ssim', 'psnr', 'l1', 'mae'], data_range=1, win_size=51, multichannel=True):
        self.data_range = data_range
        self.win_size = win_size
        self.multichannel = multichannel
        self.device = device
        self.lpips = LPIPS(
            net_type='vgg',  # choose a network type from ['alex', 'squeeze', 'vgg']
            version='0.1'  # Currently, v0.1 is supported 
        )
        for metric in metric_list:
            if metric in ['ssim', 'psnr', 'l1', 'mae', 'lpips']:
                setattr(self, metric, True)
            else:
                print('unsupport reconstruction metric: %s'%metric)


    def __call__(self, inputs, gts):
        """
        inputs: the generated image, size (b,c,w,h), data range(0, data_range)
        gts:    the ground-truth image, size (b,c,w,h), data range(0, data_range)
        """
        result = dict() 
        [b,n,w,h] = inputs.size()
        inputs = inputs.view(b*n, w, h).detach().cpu().numpy().astype(np.float32).transpose(1,2,0)
        gts = gts.view(b*n, w, h).detach().cpu().numpy().astype(np.float32).transpose(1,2,0)

        if hasattr(self, 'ssim'):
            ssim_value = compare_ssim(inputs, gts, data_range=self.data_range, 
                            win_size=self.win_size, multichannel=self.multichannel) 
            result['ssim'] = ssim_value

        if hasattr(self, 'psnr'):
            psnr_value = compare_psnr(inputs, gts, self.data_range)
            result['psnr'] = psnr_value

        if hasattr(self, 'l1'):
            l1_value = compare_l1(inputs, gts)
            result['l1'] = l1_value            

        if hasattr(self, 'mae'):
            mae_value = compare_mae(inputs, gts)
            result['mae'] = mae_value              
        return result

        if hasattr(self, 'lpips'):
            lpips_value = compare_lpips(self, inputs, gts)
            result['lpips'] = lpips_value.cpu().numpy()       


    def calculate_from_disk(self, inputs, gts, save_path=None, debug=0):
        """
            inputs: .txt files, floders, image files (string), image files (list)
            gts: .txt files, floders, image files (string), image files (list)
        """
        input_image_list = sorted(get_image_list(inputs))
        gt_image_list = sorted(get_image_list(gts))
        print(len(input_image_list))
        print(len(gt_image_list))

        psnr = []
        ssim = []
        mae = []
        l1 = []
        names = []
        lpips_ar = []

        for index in range(len(input_image_list)):
            name = os.path.basename(input_image_list[index])
            names.append(name)

            try:
                img_gt   = (imageio.imread(str(gt_image_list[index]))).astype(np.float32) / 255.0
                img_pred = (imageio.imread(str(input_image_list[index]))).astype(np.float32) / 255.0
            except:
                continue

            if debug != 0:
                plt.subplot('121')
                plt.imshow(img_gt)
                plt.title('Groud truth')
                plt.subplot('122')
                plt.imshow(img_pred)
                plt.title('Output')
                plt.show()

            psnr.append(compare_psnr(img_gt, img_pred, data_range=self.data_range))
            ssim.append(compare_ssim(img_gt, img_pred, data_range=self.data_range, 
                        win_size=self.win_size,multichannel=self.multichannel))
            mae.append(compare_mae(img_gt, img_pred))
            l1.append(compare_l1(img_gt, img_pred))
            lpips_ar.append(compare_lpips(self, img_gt, img_pred))


            if np.mod(index, 200) == 0:
                print(
                    str(index) + ' images processed',
                    "PSNR: %.4f" % round(np.mean(psnr), 4),
                    "SSIM: %.4f" % round(np.mean(ssim), 4),
                    "MAE: %.4f" % round(np.mean(mae), 4),
                    "l1: %.4f" % round(np.mean(l1), 4),
                    "LPIPS: %.4f" % (torch.mean(torch.tensor(lpips_ar))),
                )
            
        #if save_path:
            #np.savez(save_path + '/metrics.npz', psnr=psnr, ssim=ssim, mae=mae, names=names) 

        print(
            "PSNR: %.4f" % round(np.mean(psnr), 4),
            "PSNR Variance: %.4f" % round(np.var(psnr), 4),
            "SSIM: %.4f" % round(np.mean(ssim), 4),
            "SSIM Variance: %.4f" % round(np.var(ssim), 4),
            "MAE: %.4f" % round(np.mean(mae), 4),
            "MAE Variance: %.4f" % round(np.var(mae), 4),
            "l1: %.4f" % round(np.mean(l1), 4),
            "l1 Variance: %.4f" % round(np.var(l1), 4),
            "LPIPS: %.4f" % (torch.mean(torch.tensor(lpips_ar)))
        )    
        return np.mean(psnr), np.mean(ssim), np.mean(l1), np.mean(mae), (torch.mean(torch.tensor(lpips_ar)))


def get_image_list(flist):
    if isinstance(flist, list):
        return flist

    # flist: image file path, image directory path, text file flist path
    if isinstance(flist, str):
        if os.path.isdir(flist):
            flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
            flist.sort()
            return flist

        if os.path.isfile(flist):
            try:
                return np.genfromtxt(flist, dtype=np.str)
            except:
                return [flist]
    print('can not read files from %s return empty list'%flist)
    return []

def compare_l1(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.mean(np.abs(img_true - img_test))    

def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)

def compare_lpips(self, img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    return self.lpips(torch.from_numpy(img_true).unsqueeze_(0).permute(0,3,1,2), torch.from_numpy(img_test).unsqueeze_(0).permute(0,3,1,2))





