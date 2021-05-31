import visdom
import numpy
import numpy as np
import imageio as im
import torch
import datetime
from PIL import Image
from skimage import io, color
from json2html import *
class NullVisualizer(object):
    def __init__(self):
        self.name = __name__

    def append_loss(self, epoch, global_iteration, loss, mode='train'):
        pass

    def show_images(self, images, title):
        pass

class VisdomPlotVisualizer(object):
    def __init__(self, name, server="http://localhost"):
        self.visualizer = visdom.Visdom(server=server, port=8097, env=name,\
            use_incoming_socket=False)
        self.name = name
        self.server = server
        self.first_train_value = True
        self.first_test_value = True
        self.plots = {}                                
        
    def append_loss(self, epoch, global_iteration, loss, loss_name="total", mode='train'):
        plot_name = loss_name + ('_loss' if mode == 'train' else '_error')
        opts = (
            {
                'title': plot_name,
                #'legend': mode,
                'xlabel': 'iterations',
                'ylabel': loss_name
            })
        loss_value = float(loss.detach().cpu().numpy())
        if loss_name not in self.plots:
            self.plots[loss_name] = self.visualizer.line(X=numpy.array([global_iteration]),\
                Y=numpy.array([loss_value]), opts=opts)
        else:
            self.visualizer.line(X=numpy.array([global_iteration]),\
                Y=numpy.array([loss_value]), win=self.plots[loss_name], name=mode, update='append')

    
    def append_metric(self, epoch, global_iteration, loss, loss_name="total", mode='train'):
        plot_name = loss_name + ('_metric' if mode == 'train' else '_error')
        opts = (
            {
                'title': plot_name,
                #'legend': mode,
                'xlabel': 'Epochs',
                'ylabel': loss_name
            })
        loss_value = float(loss.detach().cpu().numpy())
        if loss_name not in self.plots:
            self.plots[loss_name] = self.visualizer.line(X=numpy.array([epoch]),\
                Y=numpy.array([loss_value]), opts=opts)
        else:
            self.visualizer.line(X=numpy.array([epoch]),\
                Y=numpy.array([loss_value]), win=self.plots[loss_name], name=mode, update='append')

    def config(self, **kwargs):
        # self.visualizer.text("@{6}<br>LR={0}<br>WD={1}<br>OPT:{2}<br>Photo({3})-Smooth({4})<br>MaxDepth={5}<br>"\
        #     #.format(lr, weight_decay,optimizer,photo_w,smooth_reg_w,depth_thres))
        #     .format(kwargs['lr'], kwargs['weight_decay'],kwargs['optimizer'],\
        #         kwargs['photo_w'],kwargs['smooth_reg_w'],kwargs['depth_thres'],\
        #         datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))
        self.visualizer.text(json2html.convert(json=dict(kwargs)))
        # self.visualizer.text(json2html.convert(json=locals()))

class VisdomImageVisualizer(object):
    def __init__(self, name, server="http://localhost", count=2):
        self.name = name
        self.server = server
        self.count = count

    def update_epoch(self, epoch):
        self.visualizer = visdom.Visdom(server=self.server, port=8097,\
            env=self.name + str(epoch), use_incoming_socket=False)    
    
    def show_images(self, images, title):
        b, c, h, w = images.size()        
        recon_images = images.detach().cpu()[:self.count, [2, 1, 0], :, :]\
            if c == 3 else\
            images.detach().cpu()[:self.count, :, :, :]
        opts = (
        {
            'title': title, 'width': self.count / 2 * 512,
            'height': self.count / 4 * 256
        })
        self.visualizer.images(recon_images, opts=opts,\
            win=self.name + title + "_window")

    def show_separate_images(self, images, title, iteration_counter):
        b, c, h, w = images.size()
        take = self.count if self.count < b else b
        
        recon_images = images.detach().cpu()[:take, :, :, :]\
            if c == 3 else images.detach().cpu()[:take, :, :, :]
        for i in range(take):
            #relitArray = images[i, :, :, :].cpu().detach().numpy()
            #relitArray = np.float32(relitArray)
            #relitArray = np.transpose(relitArray,(1,2,0))
            #im.imwrite("./out/_"+str(iteration_counter)+'_'+str(i)+".exr", relitArray.astype(np.float32))
            img = recon_images[i, :, :, :]
            #img = img
            opts = (
            {
                'title': title + "_" + str(i),
                 'width': w, 'height': h
            })
            self.visualizer.image((img), opts=opts,\
                win=self.name + title + "_window_" + str(i))
    
    def show_separate_images(self, images, title, iteration):
        b, c, h, w = images.size()  
        take = self.count if self.count < b else b
        images = torch.clamp(images, min=0.0, max=1.0)
        recon_images = images.detach().cpu()[:take, :, :, :]\
            if c == 3 else images.detach().cpu()[:take, :, :, :]
        for i in range(take):
            img = recon_images[i, :, :, :]
            opts = (
            {
                'title': title + "_" + str(i),
                 'width': w, 'height': h
            })
            self.visualizer.image(img, opts=opts,\
                win=self.name + title + "_window_" + str(i))
                
            #self.visualizer.image(img2, opts=opts,\
            #    win=self.name + title + "_L_window_" + str(i))

    def show_separate_images2(self, images, title):
        b, c, h, w = images.size()
        take = self.count if self.count < b else b
        images = torch.clamp(images, min=0.0, max=1.0)
        recon_images = images.detach().cpu()[:take, [2, 1, 0], :, :]\
            if c == 3 else (images.detach().cpu())[:take, :, :, :]
        for i in range(take):
            img = recon_images[i, :, :, :]
            opts = (
            {
                'title': title + "_" + str(i),
                    'width': w, 'height': h
            })
            self.visualizer.image(img, opts=opts,\
                win=self.name + title + "_window_" + str(i))

    def show_activations(self, maps, title):
        b, c, h, w = maps.size()
        maps_cpu = maps.detach().cpu()[:1, :, :, :]
        maps_cpu = maps_cpu.squeeze(0)
        for i in range(c):
            opts = (
            {
                'title': title + str(i), 'colormap': 'Viridis'
            })
            heatmap = maps_cpu[i, :, :]
            self.visualizer.heatmap(heatmap,\
                opts=opts, win=self.name + title + "_window_" + str(i))
