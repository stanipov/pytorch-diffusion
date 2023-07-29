import numpy as np
import torch
from typing import Union, Tuple

import PIL
from PIL import Image
from PIL.Image import BICUBIC, LANCZOS
import torchvision.transforms.functional as F
import torchvision.transforms.v2.functional as F2

from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset as PtDataset
from datasets import load_dataset
from torch.utils.data import random_split

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

class HUGGAN_Dataset(PtDataset):
    def __init__(self, hf_dataset, new_img_size: int = 256,
                 resample: PIL.Image.Resampling = PIL.Image.Resampling.BICUBIC,
                 flip_prob: float = 0.5):
        self.data = hf_dataset
        self.new_img_size = new_img_size
        self.resample = resample
        self.transform = transforms.Compose([
            MyLambdaPILResAspect(pil_resize, new_img_size, resample),
            v2.RandomEqualize(p=0.5),
            v2.RandomHorizontalFlip(flip_prob),
            v2.RandomVerticalFlip(flip_prob),
            v2.RandomAdjustSharpness(sharpness_factor=3, p=0.75),
            v2.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.2, hue=0.2),
            v2.AugMix(interpolation=transforms.InterpolationMode.BILINEAR, severity=3),
            SquarePad(),
            v2.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        res = self.data[idx]
        images = res['image']
        artists = res['artist']
        genre = res['genre']
        style = res['style']
        imgs = [self.transform(image) for image in images] if type(images) == list else [self.transform(images)]
        return imgs, artists, genre, style

def hash_lbls(*args):
    return [hash(item) for item in zip(*args)]

def collate_fn(batch):
    imgs, artists, genre, style = zip(*batch)
    lbls = torch.tensor(hash_lbls(*(artists, genre, style)))
    imgs = torch.stack([y[0] for y in imgs ])
    return imgs, lbls

# ------------------------------------------------ DataLoaders ---------------------------------------------------------
def set_dataloader_unet_hf(config):
    legacy = config['dataset'].get('type', 'legacy')

    # using the better dataset from HuggingFace dataset hub
    print('Will be using HuggingFace dataset')
    image_size = config['dataset']['image_size']
    root = config['dataset']['location'] # HF downloaded dataset location
    use_subset = config['dataset']['use_subset']
    batch_size = int(config['training']['batch_size'])
    dataloader_workers = int(config['training']['dataloader_workers'])
    flip_prob = config['dataset'].get('flip_prob', 0.0)
    if use_subset:
        use_subset = float(use_subset)
    img_resize = config['dataset']['img_resize']

    print('Setting the dataset')
    hf_dataset = load_dataset('huggan/wikiart', split="train", cache_dir=root)
    dataset = HUGGAN_Dataset(hf_dataset, new_img_size=image_size,
                             resample=PIL.Image.Resampling.LANCZOS, flip_prob=flip_prob)

    if use_subset:
        print(f'\t{use_subset} of the total dataset will be used')
        new_ds_len = int(len(dataset) * use_subset)
        splits = [new_ds_len, len(dataset) - new_ds_len]
        dataset, _ = random_split(dataset, splits)
        print(f'\t{len(dataset)} of images will be used')
    else:
        print(f'Using whole of {len(dataset)} images')

    num_features = []
    for key in hf_dataset.features.keys():
        if key != 'image':
            #total_lbls += hf_dataset.features[key].num_classes
            num_features.append(hf_dataset.features[key].num_classes)
    print(f'\t{sum(num_features)} classes were found')

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=dataloader_workers,
                                               pin_memory=True, collate_fn=collate_fn)
    print('Done')
    return train_loader, num_features


def set_dataloader_vq_hf(config):
    # using the better dataset from HuggingFace dataset hub

    print('Will be using HuggingFace dataset')
    image_size = config['dataset']['image_size']
    root = config['dataset']['location']  # HF downloaded dataset location
    use_subset = config['dataset']['use_subset']
    batch_size = int(config['training']['batch_size'])
    dataloader_workers = int(config['training']['dataloader_workers'])
    flip_prob = config['dataset'].get('flip_prob', 0.0)
    if use_subset:
        use_subset = float(use_subset)

    print('Setting the dataset')
    hf_dataset = load_dataset('huggan/wikiart', split="train", cache_dir=root)
    dataset = HUGGAN_Dataset(hf_dataset, new_img_size=image_size,
                             resample=PIL.Image.Resampling.LANCZOS, flip_prob=flip_prob)

    if use_subset:
        print(f'\t{use_subset} of the total dataset will be used')
        new_ds_len = int(len(dataset) * use_subset)
        splits = [new_ds_len, len(dataset) - new_ds_len]
        dataset, _ = random_split(dataset, splits)
        print(f'\t{len(dataset)} of images will be used')
    else:
        print(f'Using whole of {len(dataset)} images')

    # get total number of all features/classes
    total_lbls = 0
    for key in hf_dataset.features.keys():
        if key != 'image':
            total_lbls += hf_dataset.features[key].num_classes
    print(f'\t{total_lbls} classes were found')

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=dataloader_workers,
                                               pin_memory=True, collate_fn=collate_fn)
    print('Done')
    return train_loader


def set_dataloader_disc_hf(config):

    image_size = config['dataset']['image_size']
    root = config['dataset']['location']
    use_subset = config['dataset']['use_subset']
    batch_size = int(config['training']['batch_size'])
    flip_prob = config['dataset'].get('flip_prob', 0.0)
    if config['discriminator']['disc_train_batch']:
        batch_size = int(config['discriminator']['disc_train_batch'])

    dataloader_workers = int(config['training']['dataloader_workers'])
    if use_subset:
        use_subset = float(use_subset)
    img_resize = config['dataset']['img_resize']

    print('Setting the dataset')
    hf_dataset = load_dataset('huggan/wikiart', split="train", cache_dir=root)
    dataset = HUGGAN_Dataset(hf_dataset, new_img_size=image_size,
                             resample=PIL.Image.Resampling.LANCZOS, flip_prob=flip_prob)

    if use_subset:
        print(f'\t{use_subset} of the total dataset will be used')
        new_ds_len = int(len(dataset) * use_subset)
        splits = [new_ds_len, len(dataset) - new_ds_len]
        dataset, _ = random_split(dataset, splits)
        print(f'\t{len(dataset)} of images will be used')
    else:
        print(f'Using whole of {len(dataset)} images')

    # get total number of all features/classes
    total_lbls = 0
    for key in hf_dataset.features.keys():
        if key != 'image':
            total_lbls += hf_dataset.features[key].num_classes
    print(f'\t{total_lbls} classes were found')

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=dataloader_workers,
                                               pin_memory=True, collate_fn=collate_fn)
    print('Done')
    return train_loader