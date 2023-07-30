import numpy as np
import torch
import os, glob

import webdataset as wds

import PIL
from PIL import Image
from PIL.Image import BICUBIC, LANCZOS
import torchvision.transforms.functional as F
import torchvision.transforms.v2.functional as F2

from torchvision import transforms
from torchvision.transforms import v2

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MyLambdaPILResAspect(transforms.Lambda):
    """ Resize keeping aspect ratio """
    def __init__(self, lambd, size, method=LANCZOS):
        super().__init__(lambd)
        self.size = size
        self.method=method

    def __call__(self, img):
        return self.lambd(img, self.size, self.method)


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp + (max_wh - w) % 2, vp + (max_wh - h) % 2 )
        return F2.pad(image, padding, 0, 'constant')


def pil_resize(img: Image, size, method):
    """ PIL resize keeping aspect ratio """
    aspect_ratio = img.height / img.width
    if img.width > img.height:
        new_w = size
        new_h = int(new_w * aspect_ratio)
    if img.width <= img.height:
        new_h = size
        new_w = int(new_h / aspect_ratio)
    new_size = (new_w, new_h)

    return img.resize(size=new_size, resample=method)