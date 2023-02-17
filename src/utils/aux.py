from torchvision import utils
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def unscale_tensor(T, mean, std):
    """
    revert transforms.Normalize() operation
    """
    return T * std + mean
    

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
    
    
def plot_loss(val_loss,train_loss):
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(1,1)
    X = np.arange(1, val_loss.shape[0]+1, 1)
    l, = ax.plot(X, val_loss)
    l.set_label('Val')
    l, = ax.plot(X, train_loss)
    l.set_label('Train')
    ax.legend()
    return fig, ax
    
    
def mean_std(loader):
    """
    Finds mean and std for the whole dataset.
    
    The dataloader must have bath size to be equal 
    to the length of the dataset.
    
    Avoid any randomness (i.e. random modification in the image transforms,
    if any) or shuffling the dataset
    
    Taken from 
    https://www.binarystudy.com/2022/04/how-to-normalize-image-dataset-inpytorch.html
    """
    images = next(iter(loader))
    # shape of images = [b,c,w,h]
    # returns mean, std 
    return images.mean([0,2,3]), images.std([0,2,3])
