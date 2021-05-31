""" Module implementing various loss functions """

import torch as th


# =============================================================
# Interface for the losses
# =============================================================

class GANLoss:
    """ Base class for all losses
        @args:
            dis: Discriminator used for calculating the loss
                 Note this must be a part of the GAN framework
    """

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps):
        """
        calculate the discriminator loss using the following data
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps):
        """
        calculate the generator loss
        :param real_samps: batch of real samples
        :param fake_samps: batch of generated (fake) samples
        :return: loss => calculated loss Tensor
        """
        raise NotImplementedError("gen_loss method has not been implemented")


class LSGAN(GANLoss):
    
    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, mask):
        real_preds = self.dis(real_samps, mask, False)
        fake_preds = self.dis(fake_samps, mask, False)

        return 0.5 * (((th.mean(real_preds) - 1) ** 2)
                      + (th.mean(fake_preds)) ** 2)

    def gen_loss(self, real_samps, fake_samps, mask):
        real_preds, real_feats = self.dis(real_samps, mask, True)
        fake_preds, fake_feats = self.dis(fake_samps, mask, True)

        loss =  0.5 * ((th.mean(fake_preds) - 1) ** 2)
        
        return loss, real_feats, fake_feats



class HingeGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, mask):
        r_preds = self.dis(real_samps, mask, False)
        f_preds = self.dis(fake_samps, mask, False)

        loss = (th.mean(th.nn.ReLU()(1 - r_preds)) +
                th.mean(th.nn.ReLU()(1 + f_preds)))

        return loss

    def gen_loss(self, real_samps, fake_samps, mask):
        f_preads, f_feats = self.dis(fake_samps, mask, True)
        r_preads, r_feats = self.dis(real_samps, mask, True)
        loss = -th.mean(f_preads)

        return loss, r_feats, f_feats



class RelativisticAverageHingeGAN(GANLoss):

    def __init__(self, dis):
        super().__init__(dis)

    def dis_loss(self, real_samps, fake_samps, mask):
        # Obtain predictions
        r_preds = self.dis(real_samps, mask)
        f_preds = self.dis(fake_samps, mask)

        # difference between real and fake:
        r_f_diff = r_preds - th.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - th.mean(r_preds)

        # return the loss
        loss = (th.mean(th.nn.ReLU()(1 - r_f_diff))
                + th.mean(th.nn.ReLU()(1 + f_r_diff)))

        return loss

    def gen_loss(self, real_samps, fake_samps, mask):
        # Obtain predictions
        r_preds = self.dis(real_samps, mask)
        f_preds = self.dis(fake_samps, mask)

        # difference between real and fake:
        r_f_diff = r_preds - th.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - th.mean(r_preds)

        # return the loss
        return (th.mean(th.nn.ReLU()(1 + r_f_diff))
                + th.mean(th.nn.ReLU()(1 - f_r_diff)))