from torchvision import utils
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from torchvision.transforms import ToPILImage


def unscale_tensor(T):
    """
    Unscale a tensor from [-1,1] to [0,1]
    """
    return (T+1)/2
    
 
def save_grid_imgs(img_tensor, nrow, fname):
    """
    Saves a tensor into a grid image
    """
    out = 0
    grid_img = utils.make_grid(img_tensor.to('cpu'), nrow = nrow)
    utils.save_image(grid_img, fp = fname)
    #try:
    #    grid_img = utils.make_grid(img_tensor.to('cpu'), nrow = nrow)
    #    utils.save_image(grid_img, fp = fname)
    #except Exception as err:
    #    out = 1
    #return out


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
