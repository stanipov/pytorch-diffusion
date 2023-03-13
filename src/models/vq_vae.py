import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.decoder import Decoder2, Decoder
from src.models.encoder import Encoder2, Encoder
from src.models.conv_blocks import WeightStandardizedConv2d
from src.models.vq import VectorQuantizer
from typing import Tuple, Optional

import json
# ------------------------------------------------------------------------------------------------------------   
from typing import Tuple, Optional

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
                 up_attn: Optional[bool] = False,
                 eps: Optional[float] = 1e-6,
                 scaling_factor: float = 0.18215
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
                                 eps = eps)

        
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
                                 eps = eps)
                                 
        vq_embed_dim = vq_embed_dim if vq_embed_dim is not None else latent_dim
        self._vq = VectorQuantizer(num_vq_embeddings, vq_embed_dim, commitment_cost)
        
        self.pre_quantizer = WeightStandardizedConv2d(latent_dim, vq_embed_dim, kernel_size=1)
        self.post_quantizer = WeightStandardizedConv2d(vq_embed_dim, latent_dim, kernel_size=1)
        
        
    def encode(self, x):
        x = self._encoder(x)
        return self.pre_quantizer(x)
    
    def decode(self, z):
        loss, z_q, perplexity, encodings, encoding_indices = self._vq(z)
        z_q = self.post_quantizer(z_q)
        return self._decoder(z_q), loss, perplexity, encodings, encoding_indices
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

# ------------------------------------------------------------------------------------------------------------   
def set_VQModel(config, load = False):
    
    # Model params
    cfg = config['model']
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
    down_attn         = cfg.get('down_attn', False)
    up_mode           = cfg.get('up_mode', 'nearest')
    up_scale          = cfg.get('up_scale', 2)
    up_attn           = cfg.get('up_attn', False)
    eps               = cfg.get('eps', 1e-6)
    scaling_factor    = cfg.get('scaling_factor', 0.18215)
    
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
        scaling_factor = scaling_factor
    )

    if load:
        print(f'\tLoading the pretrined weights from\n\t{load}')
        status = _model.load_state_dict(torch.load(load))
        print(status)
    
    return _model







# ***************************************************************************************************************   

class _legacy_VQ_VAE(nn.Module):
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
        
def _legacy_set_codec(config_file, model_weights):
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
