"""
LPIPS loss with discriminator for VQ Model
Adopted from https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/losses/vqperceptual.py
"""


import torch
import torch.nn as nn
from src.losses.util import hinge_d_loss, vanilla_d_loss
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, 
                 discriminator, 
                 disc_start,
                 cfg,
                 discriminator_weight = 1.0,
                 disc_factor = 1.0,
                 disc_loss="hinge"
                ):
        super().__init__()
        
        codebook_weight = cfg.get('codebook_weight', 1.0)
        pixelloss_weight = cfg.get('pixelloss_weight', 1.0)
        perceptual_weight = cfg.get('perceptual_weight', 1.0)
        pixel_loss = cfg.get('pixel_loss', 'huber')
        disc_net = cfg.get('disc_net', 'vgg')   
        
        _loss_types = ['l1', 'l2', 'huber']
        _disc_types = ["vgg", "alex", "squeeze"]
        assert disc_loss in ["hinge", "vanilla"]
        assert pixel_loss in _loss_types, f"Expected one of '{', '.join(_loss_types)}', got '{pixel_loss}'"
        assert disc_net in _disc_types, f"Expected one of '{', '.join(_disc_types)}', got '{disc_net}'"
        self.codebook_weight = codebook_weight
        self.pixelloss_weight = pixelloss_weight
        self.perceptual_weight = perceptual_weight
        
        if pixel_loss == 'l1':
            self.pix_loss = nn.L1Loss()
        if pixel_loss == 'l2':
            self.pix_loss = nn.MSELoss()
        if pixel_loss == 'huber':
            self.pix_loss = nn.HuberLoss()
            
        if perceptual_weight: #or perceptual_weight>1e-3
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type=disc_net)
        else:
            self.lpips = None
            self.perceptual_weight = None
            
        if disc_loss == 'hinge':
            self.disc_loss = hinge_d_loss 
        elif disc_loss == 'vanilla':
            self.disc_loss = vanilla_d_loss
          
        self.disc_factor = disc_factor
        self.discriminator_weight = discriminator_weight
        self.discriminator = discriminator
        self.discriminator_iter_start = disc_start
        
    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        """
        Taken from
        https://github.com/CompVis/latent-diffusion/blob/a506df5756472e2ebaf9078affdde2c4f1502cd4/ldm/modules/losses/vqperceptual.py
        """
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]   # retain_graph=True
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]  # retain_graph=True
            d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
            d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        else:
            d_weight = torch.tensor(1.0)

        d_weight = d_weight * self.discriminator_weight
        return d_weight
        
        
    def forward(self, codebook_loss, inputs, reconstructions,
                optimizer_idx, global_step, last_layer=None, split = 'train'):

        # perceptual loss
        rec_loss = self.pix_loss(inputs.contiguous(), reconstructions.contiguous()) * self.pixelloss_weight
        if self.lpips:
            percep_loss = self.lpips(inputs.contiguous(), reconstructions.contiguous()) * self.perceptual_weight
        
        if codebook_loss is not None:
            codebook_loss = torch.tensor([0.0]).to(inputs.device)
        codebook_loss = codebook_loss*self.codebook_weight

        # LPIPS inclusion
        if self.lpips:
            nll_loss = torch.mean(rec_loss+percep_loss)
        else:
            nll_loss = torch.mean(rec_loss)

        # Generator loss
        if optimizer_idx == 0:
            d_weight = torch.tensor(0.0)
            g_loss = torch.tensor(0.0)

            if global_step > self.discriminator_iter_start:
                logits_fake = self.discriminator(reconstructions.contiguous())
                g_loss = -torch.mean(logits_fake)
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            disc_loss = self.disc_factor*d_weight*g_loss
            loss = nll_loss + disc_loss + self.codebook_weight*codebook_loss.mean()

            msg = {
                'Step' : global_step,
                'total': loss.clone().detach().mean().item(),
                'q': codebook_loss.detach().mean().item(),
                #'nll': nll_loss.detach().mean().item(),
                'rec': rec_loss.detach().mean().item(),
                'p': percep_loss.detach().mean().item() if self.lpips else 0,
                'disc': disc_loss.detach().mean().item(),
                'd_weight': d_weight.detach().item(),
                #'d_fac': disc_factor,
                'log_fake': g_loss.detach().mean().item(),
            }
            return loss, msg
        
        # Discriminator update
        if optimizer_idx == 1:
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())
            d_loss = self.disc_loss(logits_real, logits_fake)

            log = {
                'Step' : global_step,
                'disc_loss':d_loss.clone().detach().mean().item(),
                'logits_real': logits_real.detach().mean().item(),
                'logits_fake': logits_fake.detach().mean().item()}
            return d_loss, log