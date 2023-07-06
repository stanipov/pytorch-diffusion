import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.decoder import Decoder2
from src.models.encoder import Encoder2
from src.models.conv_blocks import WeightStandardizedConv2d
from src.models.vq import VectorQuantizer
#from src.train.util import partial_load_model

import json
# ------------------------------------------------------------------------------------------------------------   
from typing import Tuple, Optional, List

class VQModel(nn.Module):
    def __init__(self, 
                 img_ch: int = 3, 
                 enc_init_ch: int = 64, 
                 ch_mult: Tuple[int] = (1, 2, 4, 4),
                 grnorm_groups: int = 8, 
                 resnet_stacks: int = 2, 
                 latent_dim: int = 4, 
                 num_vq_embeddings: int = 256,
                 vq_embed_dim: Optional[int] = None,
                 commitment_cost: float = 0.25,
                 down_mode: str = 'avg',
                 down_kern: int = 2,
                 down_attn: Optional[bool] = False,
                 up_mode: str = 'bilinear',
                 up_scale: int = 2,
                 up_attn: Optional[List[int]] = [],
                 attn_heads: Optional[int] = 4,
                 attn_dim: Optional[int] = 8,
                 eps: Optional[float] = 1e-6,
                 legacy_mid: Optional[bool] = False,
                 scaling_factor: Optional[float] = 0.18215,
                 dec_tanh_out: bool = False
                 ):
        """
        VQ autoencoder
        Inputs:
            img_ch:           - image channels, 3
            enc_init_ch:      - numpber of plaines to expand from img_ch
            grnorm_groups:    - number of groups for group normalization
            resnet_stacks:    - number of resnet blocks in a stack
            num_vq_embeddings - number of codebook vectors in the VQ-VAE model
            vq_embed_dim      - dimensionalyto of the codebook vectors
            embed_dim:        - dimensionality of embeddings (latent space)
            latent_dim        - latent space dimensionality n
            commitment_cost   - how important is quantization error
            last_resnet:      - if True, adds "resnet_stacks" layers as the last layers
                              - for both encoder and decoder
            down_mode:        - if "conv", then strided convolution is used, 
                               "avg" ot "max" - average2D or maxPool2D are used respectively 
            down_kern:        - Size of the pooling kernel, has effect only if down_mode is avg or max
            down_attn:        - If True, adds attention layer before the donwsampling
            up_mode:          - If 'conv", strided transposed convolution is used, otherwise, interpolation
            up_scale          - Upscale interpolation factor, default 2
            up_attn           - If True, attention layer will be added in each upsampling block
            
        """
        super().__init__()      
        
        self._encoder = Encoder2(in_planes = img_ch,
                                 init_planes = enc_init_ch, 
                                 plains_mults = ch_mult,
                                 resnet_grnorm_groups = grnorm_groups,
                                 resnet_stacks = resnet_stacks,
                                 downsample_mode = down_mode,
                                 pool_kern = down_kern,
                                 attention = down_attn,
                                 latent_dim = latent_dim,
                                 eps = eps,
                                 legacy_mid = legacy_mid,
                                 attn_heads = attn_heads,
                                 attn_dim = attn_dim)

        
        ch_mult = tuple(reversed(list(ch_mult)))
        self._decoder = Decoder2(in_planes=latent_dim,
                                 out_planes=img_ch,
                                 init_planes=enc_init_ch,
                                 plains_divs=ch_mult,
                                 resnet_grnorm_groups=grnorm_groups,
                                 resnet_stacks=resnet_stacks,
                                 up_mode = up_mode,
                                 scale = up_scale,
                                 attention = up_attn,
                                 eps = eps,
                                 legacy_mid = legacy_mid,
                                 attn_heads = attn_heads,
                                 attn_dim = attn_dim,
                                 tanh_out = dec_tanh_out)
                                 
        vq_embed_dim = vq_embed_dim if vq_embed_dim is not None else latent_dim
        self._vq = VectorQuantizer(num_vq_embeddings, vq_embed_dim, commitment_cost)
        
        self.pre_quantizer = WeightStandardizedConv2d(latent_dim, vq_embed_dim, kernel_size=1)
        self.post_quantizer = WeightStandardizedConv2d(vq_embed_dim, latent_dim, kernel_size=1)

        self.scaling_factor = scaling_factor
        
        
    def encode(self, x, tanh = False):
        x = self._encoder(x)
        x = self.pre_quantizer(x)
        if tanh:
            x = torch.tanh(x)
        return x

    def _decode(self, z):
        loss, z_q, perplexity, encodings, encoding_indices = self._vq(z)
        z_q = self.post_quantizer(z_q)
        return self._decoder(z_q), loss, perplexity, encodings, encoding_indices
    def decode(self, z):
        _, z_q, *_ = self._vq(z)
        z_q = self.post_quantizer(z_q)
        return self._decoder(z_q)
    
    def forward(self, x, tanh = False):
        z = self.encode(x, tanh)
        return self._decode(z)
        
    def get_last_layer(self):
        return self._decoder.post_up[1].weight

# ------------------------------------------------------------------------------------------------------------   
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if classname.find('WeightStandardizedConv2d') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    if classname.find('GroupNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        

"""

def set_VQModel(config, load = False):
    
    # Model params
    if 'model' in config:
        cfg = config['model']
    elif 'vqmodel' in config:
        cfg = config['vqmodel']
    else:
        raise KeyError("Can't find model config!")

    img_ch            = cfg.get('img_channels', 3)
    enc_init_ch       = cfg.get('enc_init_channels', 64)
    ch_mult           = cfg.get('ch_mult', (1, 2, 4, 4))
    grnorm_groups     = cfg.get('grnorm_groups', 4)
    resnet_stacks     = cfg.get('resnet_stacks', 2)
    latent_dim        = cfg.get('latent_dim', 4)
    num_vq_embeddings = cfg.get('num_vq_embeddings', 256)
    vq_embed_dim      = cfg.get('vq_embed_dim', None)
    commitment_cost   = cfg.get('commitment_cost', 0.25)
    down_mode         = cfg.get('down_mode', 'max')
    down_kern         = cfg.get('down_kern', 2)
    down_attn         = cfg.get('down_attn', [])
    up_mode           = cfg.get('up_mode', 'nearest')
    up_scale          = cfg.get('up_scale', 2)
    up_attn           = cfg.get('up_attn', [])
    eps               = cfg.get('eps', 1e-6)
    scaling_factor    = cfg.get('scaling_factor', 0.18215)
    attn_heads        = cfg.get('attn_heads', None)
    attn_dim          = cfg.get('attn_dim', None)
    legacy_mid        = cfg.get('legacy_mid', False)
    dec_tanh_out      = cfg.get('dec_tanh_out', False)
    
    _model = VQModel(
        img_ch = img_ch,
        enc_init_ch= enc_init_ch,
        ch_mult = ch_mult,
        grnorm_groups = grnorm_groups, 
        resnet_stacks = resnet_stacks,
        latent_dim = latent_dim,
        num_vq_embeddings = num_vq_embeddings,
        vq_embed_dim = vq_embed_dim,
        commitment_cost = commitment_cost,
        down_mode = down_mode,
        down_kern = down_kern,
        down_attn = down_attn,
        up_mode = up_mode,
        up_scale = up_scale,
        up_attn = up_attn,
        eps = eps,
        scaling_factor = scaling_factor,
        attn_heads =attn_heads,
        attn_dim = attn_dim,
        legacy_mid = legacy_mid,
        dec_tanh_out = dec_tanh_out
    )#.apply(weights_init)

    if load:
        print(f'\tLoading the pretrained weights from\n\t{load}')
        try:
            status = _model.load_state_dict(torch.load(load), strict=True)
            print(f'\t{status}')
        except Exception as E:
            print(E)
            _model = partial_load_model(_model, load)
    
    return _model


"""

