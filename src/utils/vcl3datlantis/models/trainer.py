import torch
from torch.nn.functional import one_hot
from vcl3datlantis.misc.viz_360.visualization import VisdomPlotVisualizer
from vcl3datlantis.misc.viz_360 import *
import vcl3datlantis.misc.viz_360.utils
from vcl3datlantis.metrics.psnr import *
from vcl3datlantis.metrics.metrics import *
from vcl3datlantis.models.PanoDR.PanoDR_module import * 
from vcl3datlantis.models.PanoDR.layer import getCheckpoint

def training(args, dataloader, test_dataloader, device):

    device, visualizers, model_params = initialize(args)
    plot_viz = visualizers[0]
    img_viz = visualizers[1]
    plot_viz.config(**vars(args))

    inPaintModel = PanoDR(opt=args, device=device)    

    inPaintModel.print_networks()
    if args.load_name != '':
        print('Loading pretrained model from {}'.format(args.load_name))
        inPaintModel.load_networks(getCheckpoint(os.path.join(args.load_name, '*.pth')))
        print('Loading done.')

    rec = Reconstruction_Metrics(device)

    total_batches = len(iter(dataloader))
    limit = total_batches
    iteration = args.start_iter
    for epoch in range(args.start_epoch, args.epochs, 1):
        #if epoch != 0:
        img_viz.update_epoch(epoch)
        for (i, data) in enumerate(dataloader, 1):
            if i>limit:
                break
            inPaintModel.initData(data, epoch, iteration)
            inPaintModel.optimize_parameters()
            #inPaintModel.update_learning_rate()
        
            if iteration % args.viz_loss_every == 0:
                losses = inPaintModel.get_current_losses()
                plot_viz.append_loss(epoch, iteration, losses['G_loss_adv'], "Adversarial Generator")
                plot_viz.append_loss(epoch, iteration, losses['D_loss'], "Discriminator")
                plot_viz.append_loss(epoch, iteration, losses['G_l1_loss'], "Generator L1")
                plot_viz.append_loss(epoch, iteration, losses['G_perceptual'], "Generator Perceptual")
                plot_viz.append_loss(epoch, iteration, losses['G_style_loss'], "Generator Style")
                plot_viz.append_loss(epoch, iteration, losses['G_l1_loss'], "Generator L1")
                plot_viz.append_loss(epoch, iteration, losses['D_feat_match_loss'], "D Feature Matching")
                plot_viz.append_loss(epoch, iteration, losses['TV_loss'], "TV")
                plot_viz.append_loss(epoch, iteration, losses['G_loss'], "Generator")

            if iteration % args.viz_img_every == 0:
                viz_images = inPaintModel.get_current_visuals_tensor()
                img_viz.show_separate_images(viz_images['masked_input'].float(), "Masked Input", iteration)
                img_viz.show_separate_images(viz_images['GT_empty'].float(), "GT Empty", iteration)
                img_viz.show_separate_images(viz_images['Comp'].float(), "Completed", iteration)
                img_viz.show_separate_images(viz_images['Semantic_layout_gt'].float(), "Dense_Layout_GT", iteration)
                img_viz.show_separate_images(viz_images['Semantic_layout_pred'].float(), "Dense_Layout_Pred", iteration)
                img_viz.show_separate_images(viz_images['Gen_prediction'].float(), "Raw_Prediction", iteration)
            iteration += 1

            if epoch % args.save_model_every == 0:
                inPaintModel.save_networks(epoch + 1)

        total_test_batches = len(iter(test_dataloader))
        test_limit = total_test_batches
        test_dataloader.dataset.rng = random.Random(test_dataloader.dataset.seed)

        for (i, data) in enumerate(test_dataloader, 1):
            if i>test_limit:
                break
            inPaintModel.initData(data, epoch, iteration)
            inPaintModel.inference(epoch)
        
        psnr, ssim, mae, lpips = inPaintModel.evaluate(rec, str(epoch))
        
        plot_viz.append_metric(epoch, epoch, torch.tensor(lpips), "LPIPS")
        plot_viz.append_metric(epoch, epoch, torch.tensor(psnr), "PSNR")
        plot_viz.append_metric(epoch, epoch, torch.tensor(ssim), "SSIM")
        plot_viz.append_metric(epoch, epoch, torch.tensor(mae), "MAE")