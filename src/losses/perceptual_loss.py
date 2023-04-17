import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vgg16, VGG16_Weights

class PerceptualLoss(nn.Module):
    def __init__(self,  fp16 = torch.float16):
        super().__init__()
        vgg_w = VGG16_Weights.IMAGENET1K_V1
        self.vgg16 = vgg16(weights=vgg_w).features.eval()
        for p in self.vgg16.parameters():
            p.requires_grad = False

        self.fp16 = fp16
        self.transform = torch.nn.functional.interpolate
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def forward(self, x, target, loss_type = 'huber'):
        
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        x = (x-self.mean) / self.std
        target = (target-self.mean) / self.std
        
        if max(x.shape[-1], x.shape[-2]) > 224:
            x = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
                
        x_x = torch.cat((x, target), dim = 0)
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
            
        del x, target, vgg_recon, vgg_x, vggout
        
        return loss
