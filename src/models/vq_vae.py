import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.decoder import Decoder
from src.models.encoder import Encoder
from src.models.vq import VectorQuantizer

class VQ_VAE(nn.Module):
    def __init__(self, img_ch = 3, 
                 enc_init_ch = 64, 
                 ch_mult = [2, 4, 8], 
                 grnorm_groups = 8, 
                 resnet_stacks = 4, 
                 embed_dim = 1024, 
                 commitment_cost = 0.25,
                 last_resnet = False,
                 down_mode = 'avg',
                 down_kern = 2,
                 down_attn = False,
                 up_mode = 'bilinear',
                 up_scale = 2,
                 up_attn = False
                 ):
        """
        VQ autoencoder
        Inputs:
            img_ch: image channels, 3
            enc_init_ch: numpber of plaines to expand from img_ch
            grnorm_groups: number of groups for group normalization
            resnet_stacks: number of resnet blocks in a stack
            embed_dim: dimensionality of embeddings (latent space)
            commitment_cost: how important is quantization error
        """
        super().__init__()
        self._encoder = Encoder(in_planes=img_ch, 
                               init_planes=enc_init_ch, 
                               plains_mults=ch_mult, 
                               resnet_grnorm_groups=grnorm_groups, 
                               resnet_stacks=resnet_stacks,
                               last_resnet = last_resnet,
                               downsample_mode = down_mode,
                               pool_kern = down_kern,
                               attention = down_attn)
        ch_mult.reverse()
        self._decoder = Decoder(in_planes=enc_init_ch*max(ch_mult), 
                               out_planes=img_ch, 
                               plains_divs=ch_mult, 
                               resnet_grnorm_groups=grnorm_groups, 
                               resnet_stacks=resnet_stacks,
                               last_resnet = last_resnet,
                               up_mode = up_mode,
                               scale = up_scale,
                               attention = up_attn)
        self._vq = VectorQuantizer(enc_init_ch*max(ch_mult), embed_dim, commitment_cost)
        
    def encode(self, x):
        return self._encoder(x)
    
    def decode(self, z):
        loss, z_q, perplexity, encodings, encoding_indices = self._vq(z)
        return self._decoder(z_q), loss, perplexity, encodings, encoding_indices
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
    
        
