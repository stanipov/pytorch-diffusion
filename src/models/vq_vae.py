import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.decoder import Decoder
from src.models.encoder import Encoder
from src.models.vq import VectorQuantizer

import json
# ------------------------------------------------------------------------------------------------------------   

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
            img_ch:         - image channels, 3
            enc_init_ch:    - numpber of plaines to expand from img_ch
            grnorm_groups:  - number of groups for group normalization
            resnet_stacks:  - number of resnet blocks in a stack
            embed_dim:      - dimensionality of embeddings (latent space)
            commitment_cost:- how important is quantization error
            last_resnet:    - if True, adds "resnet_stacks" layers as the last layers
                            - for both encoder and decoder
            down_mode:      - if "conv", then strided convolution is used, 
                             "avg" ot "max" - average2D or maxPool2D are used respectively 
            down_kern:      - Size of the pooling kernel, has effect only if down_mode is avg or max
            down_attn:      - If True, adds attention layer before the donwsampling
            up_mode:        - If 'conv", strided transposed convolution is used, otherwise, interpolation
            up_scale        - Upscale interpolation factor, default 2
            up_attn         - If True, attention layer will be added in each upsampling block
            
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
# ------------------------------------------------------------------------------------------------------------   
        
def set_codec(config_file, model_weights):
    """
    Loads the pretrained model    
    
    config_file   - training configuration file
    model_weights - saved pretrained weights
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Model params   
    channels        = config['model']['channels']
    init_dim        = config['model']['init_dim']
    dim_mults       = config['model']['dim_mults']
    groups_grnorm   = config['model']['group_norm']
    resnet_stacks   = config['model']['resnet_stacks']
    embed_dim       = config['model']['embed_dim']
    commitment_cost = config['model']['commitment_cost']
    last_resnet     = config['model']['last_resnet']
    down_mode       = config['model']['down_mode']
    down_kern       = config['model']['down_kern']
    down_attn       = config['model']['down_attn']
    up_mode         = config['model']['up_mode']
    up_scale        = config['model']['up_scale']
    up_attn         = config['model']['up_attn']
    
    model = VQ_VAE(
        img_ch          = channels, 
        enc_init_ch     = init_dim,
        ch_mult         = dim_mults, 
        grnorm_groups   = groups_grnorm,
        resnet_stacks   = resnet_stacks,
        embed_dim       = embed_dim, 
        commitment_cost = commitment_cost,
        last_resnet     = last_resnet,
        down_mode       = down_mode,
        down_kern       = down_kern,
        down_attn       = down_attn,
        up_mode         = up_mode,
        up_scale        = up_scale,
        up_attn         = up_attn
    )
    model.load_state_dict(torch.load(model_weights))
    
    return model
