from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import utils, transforms
from torch.utils.data import random_split
from PIL import Image
from typing import List


class star_dataset(Dataset):
    """
    A simple wrapper to read filed from a folder.
    It requires list of paths to each image constituiting 
    a dataset (being it a training or a validation one)
    """
    def __init__(self, dataset_imgs, transforms = None, device = 'cpu'):
        self.dataset_imgs = dataset_imgs
        self.device = device
        self.transform = transforms
        
    def __len__(self):
        return len(self.dataset_imgs)
    
    def __getitem__(self, idx):     
        if self.transform:    
            return self.transform(Image.open(self.dataset_imgs[idx])).to(self.device)
        else:
             return Image.open(self.dataset_imgs[idx]).to(self.device)
            
            
def split_dataset(images: List[str], train_val:float = 0.8):
    """
    Splits list of images into lists of such 
    for training and validation.
    
    Inputs:
        images: list[str], list of full path to each image
        train_val: float
    
    Rerun:
        train_imgs: List[str], list of path to each image in train dataset
        test_imgs: List[str], list of path to each image in test dataset
    """
    
    train_size = int(train_val * len(images))
    test_size = len(images) - train_size
    train_dataset, test_dataset = random_split(images, [train_size, test_size])
    train_imgs, test_imgs = [], []

    for i in train_dataset:
        train_imgs.append(i)

    for i in test_dataset:
        test_imgs.append(i)
    
    return train_imgs, test_imgs
    
    
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
    
    
def get_star_mean_std(img_list, img_size, num_workers = 0):
    """
    Wrapper to find a mean and std for the stars dataset
    
    """
    img_transforms = transforms.Compose([
        transforms.CenterCrop(img_size),
        transforms.Resize(max(img_size)),
        transforms.ToTensor(),    
    ])
    total_dataset = star_dataset(img_list, img_transforms, device='cpu')
    total_dataloader = DataLoader(
                        dataset = total_dataset,
                        batch_size = len(total_dataset), shuffle=False, 
                        num_workers=num_workers)
    return mean_std(total_dataloader)

    
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

