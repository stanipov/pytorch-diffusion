from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder

def artbench256(root):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda t: (t * 2) - 1)])
    return ImageFolder(root = root, loader=Image.open, transform = transform)
    
    
def artbench_hires(root, image_size = False):
    if image_size:
        transform = transforms.Compose([transforms.Resize(int(np.ceil(image_size*1.25)), 
                                                            interpolation=transforms.InterpolationMode.BICUBIC, antialias = True),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Lambda(lambda t: (t * 2) - 1)]) 
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda t: (t * 2) - 1)])
    
    return ImageFolder(root = root, loader=Image.open, transform = transform)
