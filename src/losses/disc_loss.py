import torch
from torch import nn
from src.losses.util import hinge_d_loss, vanilla_d_loss, loss_fn

class DiscLoss(nn.Module):
    def __init__(self, disc_model, kind = 'hinge'):
        super().__init__()
        assert kind in ["hinge", "vanilla"]
        if kind == "hinge":
            self.disc_loss = hinge_d_loss
        elif kind == "vanilla":
            self.disc_loss = vanilla_d_loss 
        self.discriminator = disc_model
            
    def forward(self, x_real, x_recon):
        logits_real = self.discriminator(x_real.contiguous().detach())
        logits_fake = self.discriminator(x_recon.contiguous().detach())
        d_loss = self.disc_loss(logits_real, logits_fake)
        
        #log = {
        #    f'disc_loss':f'{d_loss.clone().detach().mean():>.10f}',
        #    f'logits_real':f'{logits_real.clone().detach().mean():>.5f}',
        #    f'logits_fake':f'{logits_fake.clone().detach().mean():>.5f}',    
        #}
        
        log = {
            f'disc_loss': d_loss.clone().detach().mean(),
            f'logits_real': logits_real.clone().detach().mean(),
            f'logits_fake': logits_fake.clone().detach().mean(),    
        }
        
        return d_loss, log
