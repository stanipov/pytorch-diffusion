"""
LPIPS loss with discriminator for VAE Model (aka KL Model)
Adopted from https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/losses/contperceptual.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from src.losses.util import hinge_d_loss, vanilla_d_loss
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from src.losses.focal_frequency_loss import FocalFrequencyLoss

def l1(x, y):
    return torch.abs(x-y)

def l2(x, y):
    return ((x-y)**2)#.mean()

def get_focal_loss_fn(cfg):
    """
        loss_weight (float): weight for focal frequency loss. Default: 1.0
        alpha (float): the scaling factor alpha of the spectrum weight matrix for flexibility. Default: 1.0
        patch_factor (int): the factor to crop image patches for patch-based focal frequency loss. Default: 1
        ave_spectrum (bool): whether to use minibatch average spectrum. Default: False
        log_matrix (bool): whether to adjust the spectrum weight matrix by logarithm. Default: False
        batch_matrix (bool): whether to calculate the spectrum weight matrix using batch-based statistics. Default: False

    """
    loss_weight = cfg.get('loss_weight', 1.0)
    alpha = cfg.get('alpha', 1.0)
    patch_factor = int(cfg.get('patch_factor', 1))
    ave_spectrum = cfg.get('ave_spectrum', False)
    log_matrix = cfg.get('log_matrix', False)
    batch_matrix = cfg.get('batch_matrix', False)
    return FocalFrequencyLoss(loss_weight=loss_weight,
                              alpha=alpha,
                              patch_factor=patch_factor,
                              ave_spectrum=ave_spectrum,
                              log_matrix=log_matrix,
                              batch_matrix=batch_matrix)

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

class KL_LPIPSWithDiscriminator(nn.Module):
    def __init__(self,
                 discriminator,
                 cfg
                 ):

        super().__init__()

        kl_weight = cfg.get('kl_weight', 1.0)
        pixelloss_weight = cfg.get('pixelloss_weight', 1.0)
        perceptual_weight = cfg.get('perceptual_weight', 1.0)
        pixel_loss = cfg.get('pixel_loss', 'huber')
        lpips_net = cfg.get('lpips_net', 'vgg')
        logvar_init = cfg.get('logvar_init', 0.0)

        disc_start = cfg.get('disc_start', 0)
        discriminator_weight = cfg.get('disc_weight', 1.0)
        disc_factor = cfg.get('disc_factor', 1.0)
        disc_loss = cfg.get('disc_loss_fn', 'hinge')

        self.disc_factor = disc_factor
        self.discriminator_weight = discriminator_weight
        self.discriminator = discriminator
        self.discriminator_iter_start = disc_start

        _loss_types = ['l1', 'l2', 'huber']
        _disc_types = ["vgg", "alex", "squeeze"]
        assert disc_loss in ["hinge", "vanilla"]
        assert pixel_loss in _loss_types, f"Expected one of '{', '.join(_loss_types)}', got '{pixel_loss}'"
        assert lpips_net in _disc_types, f"Expected one of '{', '.join(_disc_types)}', got '{lpips_net}'"

        self.kl_weight = kl_weight
        self.pixelloss_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight

        if pixel_loss == 'l1':
            self.pix_loss = l1
        if pixel_loss == 'l2':
            self.pix_loss = l2
        if pixel_loss == 'huber':
            self.pix_loss = nn.HuberLoss()

        self.focal_loss = None
        self.freq_factor = 0
        if cfg.get('focal_freq', None):
            if cfg['focal_freq'].get('enabled', False):
                print('Setting Focal Frequency loss')
                self.focal_loss = get_focal_loss_fn(cfg['focal_freq'])
                self.freq_factor = cfg['focal_freq'].get('loss_factor', 1.0)

        if perceptual_weight:  # or perceptual_weight>1e-3
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type=lpips_net)
        else:
            self.lpips = None
            self.perceptual_weight = None

        if disc_loss == 'hinge':
            self.disc_loss = hinge_d_loss
        elif disc_loss == 'vanilla':
            self.disc_loss = vanilla_d_loss

        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init, requires_grad=True)
        self.logvar.requires_grad = True

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        """
        Taken from
        https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/losses/vqperceptual.py
        """
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]  # retain_graph=True
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]  # retain_graph=True
            d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
            d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        else:
            d_weight = torch.tensor(1.0)

        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, weights=None):

        rec_loss = self.pix_loss(inputs.contiguous(), reconstructions.contiguous()) * self.pixelloss_weight

        # perceptual loss
        if self.lpips:
            p_loss = self.lpips(inputs.contiguous(), reconstructions.contiguous()) * self.perceptual_weight
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        # focal frequency loss
        if self.focal_loss:
            freq_loss = self.focal_loss(pred=reconstructions.contiguous().type(torch.float32), target=inputs.contiguous().type(torch.float32))*self.freq_factor
        else:
            freq_loss = torch([0])

        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:

            nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
            weighted_nll_loss = nll_loss
            if weights is not None:
                weighted_nll_loss = weights * nll_loss
            weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
            nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

            d_weight = torch.tensor(0.0)
            g_loss = torch.tensor(0.0)

            if global_step > self.discriminator_iter_start:
                logits_fake = self.discriminator(reconstructions.contiguous())
                g_loss = -torch.mean(logits_fake)
                try:
                    #d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer) # CompViz
                    d_weight = self.calculate_adaptive_weight(weighted_nll_loss, g_loss, last_layer=last_layer) # Me
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            disc_loss = self.disc_factor * d_weight * g_loss
            loss = weighted_nll_loss + disc_loss + self.kl_weight * kl_loss + freq_loss.type(inputs.dtype)

            msg = {
                'Step': global_step,
                'total': loss.clone().detach().mean().item(),
                'kl': kl_loss.detach().mean().item(),
                'fft': freq_loss.detach().item() if self.focal_loss else 0,
                #'nll': nll_loss.detach().mean().item(),
                'wnll': weighted_nll_loss.detach().mean().item(),
                'rec': rec_loss.detach().mean().item(),
                'p': p_loss.detach().mean().item() if self.lpips else 0,
                'disc': disc_loss.detach().mean().item(),
                'd_weight': d_weight.detach().item(),
                #'log_fake': g_loss.detach().mean().item(),
            }
            return loss, msg


        # Discriminator update
        if optimizer_idx == 1:
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                'Step': global_step,
                'disc_loss': d_loss.clone().detach().mean().item(),
                'logits_real': logits_real.detach().mean().item(),
                'logits_fake': logits_fake.detach().mean().item()}
            return d_loss, log


            # in case of no disc
            # TODO
            #loss = weighted_nll_loss +  self.kl_weight * kl_loss
            #msg = {
            #    'Step': global_step,
            #    'total': loss.clone().detach().mean().item(),
            #    'kl': kl_loss.detach().mean().item(),
            #    # 'logvar': self.logvar.detach().item(),
            #    'nll': nll_loss.detach().mean().item(),
            #    'wnll': weighted_nll_loss.detach().mean().item(),
            #    'rec': rec_loss.detach().mean().item(),
            #    'p': p_loss.detach().mean().item() if self.lpips else 0,
            #}
            #return loss, msg