from torchvision import utils
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from torchvision.transforms import ToPILImage
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if classname.find('WeightStandardizedConv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if classname.find('GroupNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def unscale_tensor(T):
    """
    Unscale a tensor from [-1,1] to [0,1]
    """
    return (T+1)/2
    
 
def save_grid_imgs(img_tensor, nrow, fname):
    """
    Saves a tensor into a grid image
    """
    grid_img = utils.make_grid(img_tensor, nrow=nrow)
    utils.save_image(grid_img, fp=fname)

def show_grid_tensor(x, nrow = 8):
    T2img    = ToPILImage()
    x_unsc   = unscale_tensor(x)
    grid_img = utils.make_grid(x_unsc.to('cpu'), nrow = nrow)
    return T2img(grid_img)

def get_num_params(m):
    return sum(p.numel() for p in m.parameters())
    
    
def cos_schedule(t, xmax, xmin, Tmax):
    return xmin + 0.5*(xmax-xmin)*(1 + np.cos(t/Tmax*np.pi))
    
    
def running_mean(x, N):
    return np.convolve(x, np.ones(N)/N, mode='valid')    
  
   
def get_model_mem(model):
    """
    Calculates memory consumption by the model
    """
    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    return mem_params + mem_bufs, mem_params, mem_bufs
