from torchvision import utils
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


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
    try:
        grid_img = utils.make_grid(img_tensor.to('cpu'), nrow = nrow)
        utils.save_image(grid_img, fp = fname)
    except:
        out = 1
    return out

