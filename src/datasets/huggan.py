import numpy as np
import torch

import PIL
from PIL import Image
from PIL.Image import BICUBIC, LANCZOS
import torchvision.transforms.functional as F
import torchvision.transforms.v2.functional as F2

from src.datasets.helpers import MyLambdaPILResAspect, SquarePad, pil_resize

from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import Dataset as PtDataset
from datasets import load_dataset
from torch.utils.data import random_split

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


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
            v2.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.1, hue=0.1),
            #v2.AugMix(interpolation=transforms.InterpolationMode.BILINEAR, severity=3),
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
    imgs = torch.stack([y[0] for y in imgs])
    lbls = torch.tensor([artists, genre, style]).T
    return imgs, lbls


class LUT:
    """
    Keeps track of all seen permutations of all classes (N>1).
    It can sample from seen combinations with replacement.
    """
    def __init__(self, classes: list[int]):
        self.lut = {}
        self.inv_lut = {}
        self.max_num = 0
        self.classes = classes

    def __getitem__(self, item):
        if item not in self.lut:
            self.lut[item] = self.max_num
            self.max_num += 1
        self.inv_lut[self.max_num] = item
        return torch.as_tensor(self.lut[item])

    def __len__(self):
        return len(self.lut)

    def lut(self):
        return self.lut

    def __repr__(self):
        total = 1
        for item in self.classes:
            total *= item
        msg = f'Luk-up-table for all encountered combinations for each of {len(self.classes)} number of classes.\nTotal number of combinations is expected to be {total}.\nCurrent: {self.__len__()}'
        return msg

    def sample(self, size: int):
        return torch.as_tensor(np.random.choice(list(self.inv_lut.keys()), size=size)).reshape((size,))

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

    classes = []
    for key in hf_dataset.features.keys():
        if key != 'image':
            #total_lbls += hf_dataset.features[key].num_classes
            classes.append(hf_dataset.features[key].num_classes)
    print(f'\t{len(classes)} classes were found')

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=dataloader_workers,
                                               pin_memory=False, collate_fn=collate_fn)
    print('Done')
    return train_loader, classes


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