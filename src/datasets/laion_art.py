import webdataset as wds
import os, glob

import numpy as np
import torch

import PIL
from PIL import Image
from PIL.Image import BICUBIC, LANCZOS
import torchvision.transforms.functional as F
import torchvision.transforms.v2.functional as F2

from torchvision import transforms
from torchvision.transforms import v2

from src.datasets.helpers import MyLambdaPILResAspect, SquarePad, pil_resize


def wds_dataset(url, new_img_size, resample, keep_aspect: bool = True, shuffle_shards: int = 100):
    """ Wrapper to create a WebDataset for LAION-ART """

    def identity(x):
        return x

    if keep_aspect:
        transform_laion = transforms.Compose([
            MyLambdaPILResAspect(pil_resize, new_img_size, resample),
            SquarePad(),
            v2.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)])
    else:
        transform_laion = transforms.Compose([
            MyLambdaPILResAspect(pil_resize, int(new_img_size * 1.1), resample),
            v2.CenterCrop(new_img_size),
            v2.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)])

    dataset = (
        wds.WebDataset(url)
        .shuffle(shuffle_shards)
        .decode("pil")
        .to_tuple("jpg", "json")
        .map_tuple(transform_laion, identity)
    )

    return dataset


def set_dataloader_vq_laion_art(config):
    # using the better dataset from HuggingFace dataset hub

    print('Will be using LAION Art dataset')
    image_size = config['dataset']['image_size']
    root = config['dataset']['location']
    batch_size = int(config['training']['batch_size'])
    dataloader_workers = int(config['training']['dataloader_workers'])
    keep_aspect = config['dataset'].get('keep_aspect', True)
    shuffle_shards = config['dataset'].get('shuffle_shards', -1)
    resample = config['dataset'].get('resample', 'bicubic')
    assert resample.lower() in ['bicubic', 'lanczos'], f'resample must be bicubic or lanczos, got {resample}'
    if resample.lower() == 'bicubic':
        resample = BICUBIC
    elif resample.lower() == 'lanczos':
        resample = LANCZOS

    tar_list = glob.glob(root + '/*.tar')
    print(f'Found {len(tar_list)} tar files')

    print('Setting the dataset')
    dataset = wds_dataset(tar_list, new_img_size=image_size,
                          resample=resample, keep_aspect=keep_aspect,
                          shuffle_shards=shuffle_shards)

    train_loader = torch.utils.data.DataLoader(dataset.batched(batch_size),
                                               num_workers=dataloader_workers,
                                               batch_size=None)

    print('Done')
    return train_loader


def set_dataloader_disc_laion_art(config):
    from copy import deepcopy
    cfg = deepcopy(config)

    if config['discriminator']['disc_train_batch']:
        cfg['training']['batch_size'] = int(config['discriminator']['disc_train_batch'])

    return set_dataloader_vq_laion_art(cfg)