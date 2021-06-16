import torch
from torch.nn.functional import one_hot
from vcl3datlantis.misc.viz_360.visualization import VisdomPlotVisualizer
from vcl3datlantis.misc.viz_360 import *
import vcl3datlantis.misc.viz_360.utils
from vcl3datlantis.metrics import *
import cv2
from vcl3datlantis.models.PanoDR.PanoDR_module import * #latest
from torchvision import transforms as T
from PIL import Image

def testing(args, device, dataloader=None):
    device = torch.device("cuda:" + str(args.gpu_id) if (torch.cuda.is_available() and int(args.gpu_id) >= 0) else "cpu")  
    inPaintModel = PanoDR(opt=args, device=device)
    checkpoint = torch.load(args.eval_chkpnt_folder, map_location="cuda:{}".format(args.gpu_id))
    inPaintModel.netG.load_state_dict(checkpoint)

    if args.inference == True:
        img_path = glob.glob(args.eval_path+"*img*")
        msk_path = glob.glob(args.eval_path+"*mask*")
        for i in range(len(img_path)):
            img = cv2.imread(img_path[i], cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0
            try:
                msk = cv2.imread(msk_path[i], cv2.IMREAD_UNCHANGED)[:,:,0]
            except:
                msk = cv2.imread(msk_path[i], cv2.IMREAD_UNCHANGED)
            msk = msk/255.0

            msk = torch.from_numpy(cv2.resize(msk, (args.width,args.height), interpolation=cv2.INTER_NEAREST)).unsqueeze(0).unsqueeze(0).float().to(device)
            img = torch.from_numpy(cv2.resize(img, (args.width,args.height), interpolation=cv2.INTER_CUBIC)).unsqueeze_(0).permute(0,3,1,2).float().to(device)
            inPaintModel.inference_file(img, msk, msk_path[i])
    
    else:
        rec = Reconstruction_Metrics(device)
        total_batches = len(iter(dataloader))
        limit = total_batches
        iteration = None
        epoch = None

        for (i, data) in enumerate(dataloader, 1):
            if i>limit:
                break
            inPaintModel.initData(data, epoch, iteration)
            inPaintModel.inference(epoch)
        psnr, ssim, mae, lpips = inPaintModel.evaluate(rec, str(epoch))
        print("PSNR: {}, SSIM: {}, MAE: {}, LPIPS: {} \n".format(psnr,ssim,mae,lpips))
