from torch.optim import lr_scheduler
from .layer import init_weights
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from .basemodel import BaseModel
from .basenet import BaseNet
from vcl3datlantis.losses.AdversarialLoss import adversarial_loss
from vcl3datlantis.losses.losses import *
from vcl3datlantis.losses.AdversarialLoss.getAdvLosses import *
from vcl3datlantis.models.GatedConv.network_module import *
from vcl3datlantis.method import spherical as S360 
from .PanoDR_networks import *
from vcl3datlantis.losses.featureMatching import FeatureMatchingLoss

class InpaintingModel(BaseModel):
    def __init__(self, act=F.elu, opt=None, device=None):
        super(InpaintingModel, self).__init__()
        self.opt = opt
        self.init(opt)
        self.device = device
        self.netG = GatedGenerator(self.opt, self.device).to(self.device)
        init_weights(self.netG, init_type=self.opt.init_type)
        
        if self.opt.structure_model != "":
            ckpnt = torch.load(opt.segmentation_model_chkpnt)
            self.netG.structure_model.load_state_dict(ckpnt)
            self.netG.structure_model.to(self.device)

            print("Freezing Layout segmentation network's weights\n")
            for param in self.netG.structure_model.parameters():
                param.requires_grad = False

        norm_layer = get_norm_layer()

        self.model_names = ['D', 'G']
        if self.opt.phase == 'test':
            return

        self.netD = Discriminator(self.opt.in_d_channels).to(self.device)
        init_weights(self.netD, init_type=self.opt.init_type)


        self.optimizer_D = None
        self.optimizers = []
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(self.opt.b1, self.opt.b2))
        self.optimizers = self.optimizers + [self.optimizer_G]
        
        self.optim_segm = torch.optim.Adam(self.netG.structure_model.parameters(), opt.lr)
        self.optimizers = self.optimizers + [self.optim_segm]
        self.optimizer_D = torch.optim.Adam(filter(lambda x: x.requires_grad, self.netD.parameters()), lr=opt.lr_D,
                                            betas=(self.opt.b1, self.opt.b2))     

        self.VGG_features = VGG19().to(self.device)
        self.style_loss = StyleLoss().to(self.device)
        self.perceptual_loss = PerceptualLoss().to(self.device)
        self.FeatMatch = FeatureMatchingLoss()

        self.images = None
        self.mask = None
        self.gt_label_one_hot = None
        self.gt_empty = None
        self.inverse_mask = None
        self.masked_input = None
        self.structure_model_output = None
        self.structure_model_output_soft = None

        self.D_real_feats, self.D_fake_feats = None, None
        self.comp = None
        self.layout = None
        self.D_feat_match_loss = None
        comp_feats, gt_feats = None, None
        self.G_l1_loss, self.G_style_loss, self.G_perceptual, self.G_loss, self.D_loss, self.G_loss_adv, self.G_style_patch_loss, self.G_style_gl_loss, self.bce = None,  None,  None,  None, None, None, None, None, None
        self.coarse_percept= None
        self.TV_loss = None
        self.epoch = None
        self.iteration = None
        self.out = None
        self.second_in = None
        self.attention_weights_256 = S360.weights.theta_confidence(                
        S360.grid.create_spherical_grid(self.opt.width)).to(self.device)

        self.optimizers = self.optimizers + [self.optimizer_D]
        self.schedulers = []
        for optimizer in self.optimizers:
            self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, [self.opt.milestone_1, self.opt.milestone_2], self.opt.lr_gamma))
        self.advLoss = getAdvLoss(self.netD, self.opt.adv_loss_type)

    def get_current_learning_rate(self):
        return self.optimizers[0].param_groups[0]['lr']

    def update_learning_rate(self):
        for schedular in self.schedulers:
            schedular.step()

    def initData(self, data, epoch, iteration):
        self.mask = data["mask"].to(self.device)
        self.f_name = data["img_path"][0]
        self.images = data["img"].to(self.device)
        self.gt_label_one_hot = data['label_one_hot'].to(self.device)
        self.mask = data["mask"].to(self.device)    #0 in masked area
        self.gt_empty = data["img_gt"].to(self.device)
        self.inverse_mask = 1 - data["mask"].to(self.device)   #1 in masked area
        self.img_path = data["img_path"]
        self.sem_layout = data["label_semantic"].to(self.device)
        self.masked_input = self.images * self.mask + self.inverse_mask
        self.mask_patch_gt = self.gt_empty * self.inverse_mask 
        self.epoch = epoch
        self.iteration = iteration

    def forward_G(self):  
        self.G_loss_adv, self.D_real_feats, self.D_fake_feats = self.advLoss.gen_loss(self.gt_empty, self.out, self.inverse_mask)
        self.G_loss_adv = self.opt.lambda_adv*self.G_loss_adv

        self.comp_feats = self.VGG_features(self.out)
        self.comp_feats_patch = self.VGG_features(self.comp)
        self.gt_feats = self.VGG_features(self.gt_empty)

        self.G_perceptual = self.opt.lambda_perceptual * (self.perceptual_loss(self.comp_feats, self.gt_feats)) 
        self.G_style_loss = self.opt.lambda_style*self.style_loss(self.comp_feats, self.gt_feats)  
        self.G_style_patch_loss = self.opt.lambda_style_patch*self.style_loss(self.comp_feats_patch, self.gt_feats)
        self.G_style_loss +=self.G_style_patch_loss
        self.D_feat_match_loss = self.opt.lambda_d_match * self.FeatMatch(self.D_fake_feats, self.D_real_feats) 
        self.G_l1_loss = (self.attention_weights_256 *(self.opt.lambda_l1 * torch.abs(self.out-self.gt_empty))).mean()
        self.TV_loss = self.opt.lambda_tv * TV_loss((self.inverse_mask*self.out)) 

        self.G_loss = self.G_l1_loss + self.G_loss_adv + self.G_perceptual + self.D_feat_match_loss + self.G_style_loss + self.TV_loss 

    def forward_D(self):
        self.D_loss = self.advLoss.dis_loss(self.gt_empty, self.out.detach(), self.inverse_mask)

    def backward_G(self):
        self.G_loss.backward()

    def backward_D(self):
        self.D_loss.backward(retain_graph=True)

    def optimize_parameters(self):
        _, self.out, self.structure_model_output, self.structure_model_output_soft  = self.netG(self.images, self.inverse_mask, self.masked_input, self.device, self.opt.use_sean)
        self.mask_patch_pred = self.out * self.inverse_mask
        self.comp = self.out * self.inverse_mask + (self.images * self.mask)

        for p in self.netD.parameters():
            p.requires_grad = False

        self.optim_segm.zero_grad()
        self.optimizer_G.zero_grad()
        self.forward_G()
        self.backward_G()
        self.optimizer_G.step()
        self.optim_segm.step()

        for p in self.netD.parameters():
            p.requires_grad = True

        for i in range(self.opt.D_max_iters):
            self.optimizer_D.zero_grad()
            self.forward_D()
            self.backward_D()
            self.optimizer_D.step()
        
    def get_current_losses(self):
        _loss = {}
        if self.opt.pretrain_network == 0:
            _loss.update({'G_loss_adv':        self.G_loss_adv.detach(),
                      'G_loss':        self.G_loss.detach(),
                      'G_style_loss':  self.G_style_loss.detach(),
                      'G_perceptual': self.G_perceptual.detach(),
                      'G_l1_loss':     self.G_l1_loss.detach(),
                      'D_feat_match_loss': self.D_feat_match_loss,
                      'TV_loss':     self.TV_loss.detach(),
                      'D_loss':            self.D_loss.detach()})
        return _loss

    def get_current_visuals(self):
        return {'input':     self.masked_input.cpu().detach().numpy(), 'gt': self.gt.cpu().detach().numpy(),
                'completed': self.comp.cpu().detach().numpy()}

    def get_current_visuals_tensor(self):
        if self.opt.structure_model == "unet":
            semantic_prediction =  torch.softmax(self.structure_model_output, dim=1).cpu().detach()

        return {'Comp': self.comp.cpu().detach(),
        'masked_input' : self.masked_input.cpu().detach(),
        'GT_empty' : self.gt_empty.cpu().detach(),  
        'Semantic_layout_gt': self.gt_label_one_hot.cpu().detach(),
        'Semantic_layout_pred': semantic_prediction,
        "Gen_prediction": self.out.cpu().detach()
        }
        

    def evaluate(self, rec, epoch):
        metrics_writer = self.opt.results_path+str(self.opt.name)+".txt"
        psnr, ssim, l1, mae, lpips = rec.calculate_from_disk(self.opt.pred_results_path, self.opt.gt_results_path, save_path=self.opt.results_path)
        f = open(metrics_writer, "a")
        f.write("PSNR,{},SSIM,{},L1,{},MAE,{},LPIPS,{},Epoch,{}" .format(psnr, ssim, l1, mae, lpips, str(epoch))+"\n")
        f.close()
        return psnr, ssim, mae, lpips


    def inference(self, epoch):
        _, out , self.structure_model_output, self.structure_model_output_soft = self.netG(self.images, self.inverse_mask, self.masked_input,  self.device, self.opt.use_sean)
    
        ret =  out * self.inverse_mask + self.images * (self.mask)
        ret_masked = ret * self.inverse_mask
        ret = ret.squeeze_(0).permute(1,2,0).cpu().detach().numpy() 
        ret_masked = ret_masked.squeeze_(0).permute(1,2,0).cpu().detach().numpy() 
        
        gt_img_masked = self.gt_empty * self.inverse_mask 
        gt_img_masked = gt_img_masked.squeeze_(0).permute(1,2,0).cpu().detach().numpy() 

        _path = self.img_path[0].replace(self.opt.test_path,"")
        _path = _path.replace("\\", "/")
        f_name = "_".join(_path.split("/"))+".png" 
        pred_path = self.opt.pred_results_path+f_name
        cv2.imwrite(pred_path, (cv2.cvtColor(ret, cv2.COLOR_RGB2BGR))*255)
        gt_path = self.opt.gt_results_path+f_name
        gt_img = self.gt_empty * self.inverse_mask + self.images * self.mask
        gt_img = gt_img.squeeze_(0).permute(1,2,0).cpu().detach().numpy() 
        cv2.imwrite(gt_path, (cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR))*255)

    def inference_file(self, images, mask, f_name):

        result_path = os.path.join(self.opt.eval_path, "output/")
        os.makedirs(result_path, exist_ok=True)

        self.f_name = None
        self.images = images/255.0
        self.inverse_mask = mask/255.0
        self.mask = (1.0-self.inverse_mask)
        self.gt_empty = self.images
        masked_input = (self.images * self.mask) + self.inverse_mask
        _, out , self.structure_model_output, self.structure_model_output_soft = self.netG(self.images, self.inverse_mask, masked_input,  self.device, self.opt.use_sean)
        ret =  out * self.inverse_mask + (self.images * self.mask)
        ret_masked = ret * self.inverse_mask
        ret = ret.squeeze_(0).permute(1,2,0).cpu().detach().numpy() 
        raw_ret = out.squeeze_(0).permute(1,2,0).cpu().detach().numpy() 
        ret_masked = ret_masked.squeeze_(0).permute(1,2,0).cpu().detach().numpy() 
        gt_img_masked = self.gt_empty * self.inverse_mask 
        gt_img_masked = gt_img_masked.squeeze_(0).permute(1,2,0).cpu().detach().numpy() 

        pred_path = result_path + "diminished_" + os.path.basename(f_name)
        raw_pred_path = result_path + "raw_pred_" + os.path.basename(f_name)
        layout_path = result_path + "layout_" + os.path.basename(f_name)
        
        masked_input_np = masked_input.squeeze_(0).permute(1,2,0).cpu().detach().numpy()
        cv2.imwrite(pred_path, (cv2.cvtColor(ret, cv2.COLOR_RGB2BGR))*255)
        cv2.imwrite(result_path+"masked.png", (cv2.cvtColor(masked_input_np, cv2.COLOR_RGB2BGR))*255)
        cv2.imwrite(raw_pred_path, (cv2.cvtColor(raw_ret, cv2.COLOR_RGB2BGR))*255)

        _layout = self.structure_model_output_soft.squeeze_(0).permute(1,2,0).cpu().detach().numpy() 
        a=np.argmax(_layout, axis=2)
        z=np.zeros((256,512,3))
        z[a==0] = (255,0,0);z[a==1] = (255,255,255);z[a==2] = (0,0,255)
        z=z.astype(np.float32)
        layout_path = pred_path.replace(".png", "_layout.png")
        cv2.imwrite(layout_path, z*255)
        