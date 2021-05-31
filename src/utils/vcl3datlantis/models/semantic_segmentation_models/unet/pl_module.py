import pytorch_lightning as pl
import torch.nn as nn
import torch
from vcl3datlantis.misc.visualization import VisdomVisualizer
from vcl3datlantis.misc.spatial_gradient import gradient

EPS = 1e-6
#slightly modified
def get_IoU(outputs : torch.Tensor, labels : torch.Tensor):
    outputs = outputs.int()
    labels = labels.int()
    # Taken from: https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy
    intersection = (outputs & labels).float().sum((-1, -2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((-1, -2))  # Will be zero if both are 0

    iou = (intersection + EPS) / (union + EPS)  # We smooth our devision to avoid 0/0

    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    # return thresholded.mean()  # Or thresholded.mean() if you are interested in average across the batch
    return iou.mean(dim = (-1,-2)).sum()

def gradient_loss(t1 : torch.Tensor, t2 : torch.Tensor):
    dx1,dy1 = gradient(t1)
    dx2,dy2 = gradient(t2)

    d1 = (dx1 ** 2) + (dy1 ** 2)
    d2 = (dx2 ** 2) + (dy2 ** 2)

    loss = torch.abs(d1 - d2)
    #dot = torch.einsum("bchw,bchw -> bhw", d1, d2)
    #t = True
    return loss.mean(), loss.sum(1).unsqueeze(1)

class StructuralSemanticTrainer(pl.LightningModule):
    def __init__(self, 
                model : nn.Module,
                optimizer,
                experiment_name : str,
                loss_log_period : int = 10,
                images_log_period : int = 20,
                overfit_one_batch : bool = False,
                masked_input : bool = True
                ):
        super().__init__()
        self.model = model
        self.viz = VisdomVisualizer(experiment_name)
        self.overfit_one_batch = overfit_one_batch
        self.loss_log_period , self.images_log_period = loss_log_period, images_log_period
        self.critetion = nn.BCEWithLogitsLoss()
        self.iteration = 0
        self.validation_iter = 0
        self.optimizer = optimizer
        self.masked_input = masked_input

    def training_step(self,batch,bid):
        while True:#added for the case of overfitting one batch
            self.iteration += 1
            mask = batch["mask"]
            img = batch["img"]
            if self.masked_input:
                masked = mask * img + (1 - mask) * torch.ones_like(img)
            else:
                masked = img
            y = self.model(masked)
            y_softmax = torch.softmax(y, dim = 1)
            loss_bce = self.critetion(y, batch["label_one_hot"])
            loss_grad, loss_grad_dense = gradient_loss(y_softmax, batch["label_one_hot"])
            loss = loss_bce + 10 * loss_grad
            if self.iteration % self.loss_log_period == 0:
                self.viz.append_loss(self.current_epoch, self.iteration, loss_bce, loss_name = "BCEWithLogitsLoss")
                self.viz.append_loss(self.current_epoch, self.iteration, loss_grad, loss_name = "GradientLoss")
                self.viz.append_loss(self.current_epoch, self.iteration, loss, loss_name = "Total")
            if self.iteration % self.images_log_period == 0:
                self.viz.show_images(masked , "input")
                self.viz.show_map(loss_grad_dense, "gradient_loss")
                self.viz.show_images(y_softmax , "prediction")
                self.viz.show_images(batch["label_one_hot"] , "label_one_hot")

            if not self.overfit_one_batch:
                break
            else:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return {"loss" : loss}

    def on_validation_epoch_start(self):
        self.miou = 0
        self.validation_size = 0


    def validation_step(self, batch, bid):
        self.validation_iter += 1
        self.validation_size += batch["mask"].shape[0]
        mask = batch["mask"]
        img = batch["img"]
        masked = mask * img + (1 - mask) * torch.ones_like(img)
        y = self.model(masked)

        y_max = torch.argmax(y, dim = 1 , keepdim=True)

        self.miou += get_IoU(y_max, batch["label_semantic"])  
        if self.validation_iter % self.images_log_period == 0:
            self.viz.show_images(masked , "input_validation")
            self.viz.show_images(torch.softmax(y, dim = 1) , "prediction_validation")
            self.viz.show_images(batch["label_one_hot"] , "label_one_hot_validation")

    def validation_epoch_end(self, outputs):
        self.viz.append_loss(0,self.current_epoch,self.miou.float() / self.validation_size, "miou", "validation") 


    def configure_optimizers(self):
        return self.optimizer
