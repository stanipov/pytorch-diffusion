import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vgg16, VGG16_Weights

class PerceptualLoss(nn.Module):
    def __init__(self, device = 'cuda', fp16 = torch.float16):
        super().__init__()
        vgg_w = VGG16_Weights.IMAGENET1K_V1
        self.vgg16 = vgg16(weights=vgg_w).features
        for p in self.vgg16.parameters():
            p.requires_grad = False
        self.vgg16.to(device)
        
        self.device = device
        self.fp16 = fp16
        
    def forward(self, x, x_recon, loss_type = 'huber'):
        
        x_x = torch.cat((x, x_recon), dim = 0).to(self.device)
        sep = x.shape[0]
        
        if self.fp16:        
            with torch.cuda.amp.autocast(dtype = self.fp16):
                vggout = self.vgg16(x_x)
        else:
            vggout = self.vgg16(x_x)
        
        vgg_recon = vggout[sep:,]
        vgg_x = vggout[:sep,]
        
        if loss_type == 'l1':
            if self.fp16:
                with torch.cuda.amp.autocast(dtype = self.fp16):
                    loss = F.l1_loss(vgg_x, vgg_recon)
            else:
                loss = F.l1_loss(vgg_x, vgg_recon)

        elif loss_type == 'l2':
            if self.fp16:
                with torch.cuda.amp.autocast(dtype = self.fp16):
                    loss = F.mse_loss(vgg_x, vgg_recon)
            else:
                loss = F.mse_loss(vgg_x, vgg_recon)

        elif loss_type == "huber":
            if self.fp16:
                with torch.cuda.amp.autocast(dtype = self.fp16):        
                    loss = F.smooth_l1_loss(vgg_x, vgg_recon)
            else:
                loss = F.smooth_l1_loss(vgg_x, vgg_recon)
        else:
            raise NotImplementedError()
        
        return loss
