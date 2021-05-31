import torch
from vcl3datlantis.losses.AdversarialLoss import adversarial_loss

def getAdvLoss(netD, type_adv_loss):

    if type_adv_loss == 'RelativisticAverageHingeGAN':
        adv_loss = adversarial_loss.RelativisticAverageHingeGAN(netD)
    elif type_adv_loss == 'HingeGAN':
        adv_loss = adversarial_loss.HingeGAN(netD)
    elif type_adv_loss == 'LSGAN':
        adv_loss = adversarial_loss.LSGAN(netD)
    return adv_loss