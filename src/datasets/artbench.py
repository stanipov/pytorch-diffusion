import numpy as np
import torch
from typing import Union, Tuple

from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
# ==================================================================================================
#
#                           Datasets
#
# ==================================================================================================
def artbench256(root):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda t: (t * 2) - 1)])
    return ImageFolder(root = root, loader=Image.open, transform = transform)

def im_dataset(root,
               resize: bool = False,
               image_size: Union[int, Tuple[int,int], None] = None ):
    """
    ImageFolder wrapper to resize/leave as-is the images
    """
    if resize and image_size:
        transform = transforms.Compose([transforms.Resize(image_size,
                                                          interpolation=transforms.InterpolationMode.BICUBIC,
                                                          antialias=True),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda t: (t * 2) - 1)])
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda t: (t * 2) - 1)])
    return ImageFolder(root = root, loader=Image.open, transform = transform)

    
    
def artbench_hires(root, image_size = False):
    if image_size:
        transform = transforms.Compose([transforms.Resize(image_size,
                                        interpolation=transforms.InterpolationMode.BICUBIC,
                                        antialias = True),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda t: (t * 2) - 1)]) 
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda t: (t * 2) - 1)])
    
    return ImageFolder(root = root, loader=Image.open, transform = transform)
# ==================================================================================================
#
#                           Loaders
#
# ==================================================================================================
def set_dataloader_unet(config):
    # Dataset params
    if 'dataset' in config:
        image_size = config['dataset']['image_size']
        root = config['dataset']['location']
        use_subset = config['dataset']['use_subset']
        batch_size = int(config['training']['batch_size'])
        dataloader_workers = int(config['training']['dataloader_workers'])
        if use_subset:
            use_subset = float(use_subset)
        img_resize = config['dataset']['img_resize']

    print('Setting the dataset')
    if img_resize and image_size > 256:
        print(f'Using the original dataset with rescaling to {image_size} pix')
        dataset = im_dataset(root, resize=img_resize, image_size=image_size)
    else:
        print('Using CIFAR10-like dataset of 256 pix')
        dataset = im_dataset(root, resize=img_resize, image_size=image_size)
    num_classes = len(dataset.classes)

    if use_subset:
        print(f'\t{use_subset} of the total dataset will be used')
        new_ds_len = int(len(dataset) * use_subset)
        splits = [new_ds_len, len(dataset) - new_ds_len]
        dataset, _ = random_split(dataset, splits)
        print(f'\t{len(dataset)} of images will be used')
    else:
        print(f'Using whole of {len(dataset)} images')
    print(f'\t{num_classes} classes were found')

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=dataloader_workers,
                                               pin_memory=True)
    print('Done')
    return train_loader, num_classes



def set_dataloader_vq(config):
    # Dataset params
    if 'dataset' in config:
        image_size         = config['dataset']['image_size']
        root               = config['dataset']['location']
        use_subset         = config['dataset']['use_subset']
        batch_size         = int(config['training']['batch_size'])
        dataloader_workers = int(config['training']['dataloader_workers'])
        if use_subset:
            use_subset = float(use_subset)
        img_resize = config['dataset']['img_resize']
    
    print('Setting the dataset')
    if img_resize and image_size > 256:
        print(f'Using the original dataset with rescaling to {image_size} pix')
        dataset = im_dataset(root, resize=img_resize, image_size=image_size)
        #dataset = artbench_hires(root, image_size=image_size)
    else:
        print('Using CIFAR10-like dataset of 256 pix')
        dataset = im_dataset(root, resize=img_resize, image_size=image_size)
        #dataset = artbench256(root)
    num_classes = len(dataset.classes)

    if use_subset:
        print(f'\t{use_subset} of the total dataset will be used')
        new_ds_len = int(len(dataset)*use_subset)
        splits = [new_ds_len, len(dataset) - new_ds_len]
        dataset, _ = random_split(dataset, splits) 
        print(f'\t{len(dataset)} of images will be used')
    else:
        print(f'Using whole of {len(dataset)} images')
    print(f'\t{num_classes} classes were found')

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=dataloader_workers,
                                              pin_memory = True)
    print('Done')
    return train_loader
    
    
def set_dataloader_disc(config):
    # Dataset params
    if 'dataset' in config:
        image_size         = config['dataset']['image_size']
        root               = config['dataset']['location']
        use_subset         = config['dataset']['use_subset']
        batch_size         = int(config['training']['batch_size'])
        if config['discriminator']['disc_train_batch']:
            batch_size = int(config['discriminator']['disc_train_batch']) 
            
        dataloader_workers = int(config['training']['dataloader_workers'])
        if use_subset:
            use_subset = float(use_subset)
        img_resize = config['dataset']['img_resize']
    
    print('Setting the dataset')
    if img_resize and image_size > 256:
        print(f'Using the original dataset with rescaling to {image_size} pix')
        dataset = im_dataset(root, resize = img_resize, image_size = image_size)
        #dataset = artbench_hires(root, image_size=image_size)
    else:
        print('Using CIFAR10-like dataset of 256 pix')
        dataset = im_dataset(root, resize=img_resize, image_size=image_size)
        #dataset = artbench256(root)
    num_classes = len(dataset.classes)

    if use_subset:
        print(f'\t{use_subset} of the total dataset will be used')
        new_ds_len = int(len(dataset)*use_subset)
        splits = [new_ds_len, len(dataset) - new_ds_len]
        dataset, _ = random_split(dataset, splits) 
        print(f'\t{len(dataset)} of images will be used')
    else:
        print(f'Using whole of {len(dataset)} images')
    print(f'\t{num_classes} classes were found')

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=dataloader_workers,
                                              pin_memory = True)
    print('Done')
    return train_loader
