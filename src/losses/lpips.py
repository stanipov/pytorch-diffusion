import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def dummy_loss(t1, t2):
    return torch.tensor([0.0]).to(t1.device)

def scale_tensor_11(tensor):
    #tmin = tensor.min().item()
    #tmax = tensor.max().item() 
    tmin = tensor.min(dim = 0, keepdim = True).values
    tmax = tensor.max(dim = 0, keepdim = True).values
    return (tensor - tmin)/(abs(tmax-tmin)) * 2 -1
    


class LPIPS_VQ_loss(nn.Module):
    def __init__(self, codebook_weight=1.0, 
                 pixelloss_weight=1.0, perceptual_weight=1.0, 
                 pixel_loss='huber',
                 disc_net = 'vgg'):
        super().__init__()
        _loss_types = ['l1', 'l2', 'huber']
        _disc_types = ["vgg", "alex", "squeeze"]
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
            self.lpips = dummy_loss
            self.perceptual_weight = 0
               
    def forward(self, inputs, recons, codebook_loss):
        recon_loss = self.pix_loss(inputs.contiguous(), recons.contiguous()) * self.pixelloss_weight
        percep_loss = self.lpips(inputs.contiguous(), recons.contiguous()) * self.perceptual_weight
        
        if not codebook_loss:
            codebook_loss = torch.tensor([0.0]).to(inputs.device)
        codebook_loss = codebook_loss*self.codebook_weight
        
        loss = recon_loss + percep_loss + codebook_loss
        
        msg = {
            'total': f'{loss.item():>.5f}',
            'recon': f'{recon_loss.item():>.5f}',
            'percep': f'{percep_loss.item():>.5f}',
            'quant': f'{codebook_loss.item():>.5f}', 
        }
        
        return loss, msg
        
        
def init_lpips_loss(cfg):
    codebook_weight = cfg.get('codebook_weight', 1.0)
    pixelloss_weight = cfg.get('pixelloss_weight', 1.0)
    perceptual_weight = cfg.get('perceptual_weight', 1.0)
    pixel_loss = cfg.get('pixel_loss', 'huber')
    disc_net = cfg.get('disc_net', 'vgg')   
    
    return LPIPS_VQ_loss(codebook_weight=codebook_weight, 
                         pixelloss_weight=pixelloss_weight,
                         perceptual_weight=perceptual_weight, 
                         pixel_loss=pixel_loss,
                         disc_net = disc_net)
