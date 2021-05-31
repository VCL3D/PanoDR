import torch
from psnr import *
from fid import *
import lpips
import pytorch_ssim
from torch.autograd import Variable

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='script to compute all statistics')
    parser.add_argument('--input_path', default='/..gt/', type=str)
    parser.add_argument('--fid_gt_path', default='eval_out/', type=str)                 #Generated Images
    parser.add_argument('--fid_real_path', default='/eval_gt/', type=str)     #GT images
    args = parser.parse_args()

    #https://github.com/richzhang/PerceptualSimilarity
    #net='alex', or 'vgg' or 'vgg16' or 'squeeze'
    #install via pip: install lpips
    lpips_loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization
    PSNR_loss = PSNR()
    fid = FID()
    ssim_loss = pytorch_ssim.SSIM(window_size = 11)
    #rec = Reconstruction_Metrics()
    img0 = Variable(torch.rand(1, 3, 64, 64))
    img1 = Variable(torch.rand(1, 3, 64, 64))
    #d = lpips_loss_fn_vgg(img0, img1)
    #https://github.com/Po-Hsun-Su/pytorch-ssim
    lpips_loss_fn_vgg(img0, img1)
    ssim_loss(img0, img1)
    PSNR_loss(img0, img1)
    fid.calculate_from_disk(args.fid_gt_path, args.fid_real_path)
