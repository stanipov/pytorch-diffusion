from torch import nn
import torch
from functools import partial

from src.models.helpers import exists, default, PreNorm, Residual
from src.models.conv_blocks import ResnetBlock, Downsample, WeightStandardizedConv2d
from src.models.attention import LinearAttention

# =================================================================================================
class Encoder2(nn.Module):
    def __init__(self, 
                 in_planes = 3,
                 init_planes = 64, 
                 plains_mults = (1, 2, 4, 8),
                 resnet_grnorm_groups = 4,
                 resnet_stacks = 2,
                 downsample_mode = 'avg',
                 pool_kern = 2,
                 attention = [],
                 attn_heads = None,
                 attn_dim = None,
                 latent_dim = 4,
                 eps = 1e-6,
                 legacy_mid = False
                ):
        super().__init__()
        
        if not attn_heads:
            attn_heads = 4
        if not attn_dim:
            attn_dim = 32
           
        dims = [init_planes, *map(lambda m: init_planes * m, plains_mults)] 
        in_out = list(zip(dims[:-1], dims[1:]))
        
        conv_unit = partial(ResnetBlock, groups=resnet_grnorm_groups)
        self.init_conv = nn.Conv2d(in_planes, init_planes, 3, padding = 1)
                
        _layer = []
        for ind, (dim_in, dim_out) in enumerate(in_out):   
            is_last = ind == len(in_out) - 1
            for i in range(resnet_stacks):
                _layer.append(conv_unit(dim_in, dim_in))
            if dim_in in attention or ind in attention:
                _layer.append(Residual(PreNorm(dim_in, LinearAttention(dim_in, attn_heads, attn_dim))))
            if is_last:
                _down = WeightStandardizedConv2d(in_channels=dim_in, out_channels=dim_out, 
                                                 kernel_size=3, padding = 1, stride = 1)
            else:
                _down = Downsample(dim_in, dim_out, downsample_mode, pool_kern)
            _layer.append(_down)
        self.downsample = nn.Sequential(*_layer)
            
        # legacy midblock
        if legacy_mid:
            _layer = []
            for i in range(resnet_stacks):
                _layer.append(conv_unit(dim_out, dim_out))
        else:
            _layer = []
            _layer.append(conv_unit(dim_out, dim_out))
            _layer.append(Residual(PreNorm(dim_out, LinearAttention(dim_out, attn_heads, attn_dim))))
            _layer.append(conv_unit(dim_out, dim_out))
        self.mid_block = nn.Sequential(*_layer)
       
        self.post_enc = nn.Sequential( 
            nn.GroupNorm(num_channels=dim_out, num_groups = resnet_grnorm_groups, eps=eps),
            nn.SiLU(),
            WeightStandardizedConv2d(in_channels=dim_out, out_channels=latent_dim, kernel_size=3, padding=1),
            #nn.Tanh()
        )
                
        
    def forward(self, y):
        y = self.init_conv(y)
        y = self.downsample(y)
        y = self.mid_block(y)
        y = self.post_enc(y)
        return y
# =================================================================================================
# legacy code

class Encoder(nn.Module):
    def __init__(self, in_planes = 3,
                init_planes = 64, 
                plains_mults = (1, 2, 4, 8), 
                resnet_grnorm_groups = 4,
                resnet_stacks = 2,
                last_resnet = False,
                downsample_mode = 'avg',
                pool_kern = 2,
                attention = False
                ):
        super().__init__()
           
        dims = [init_planes, *map(lambda m: init_planes * m, plains_mults)] 
        in_out = list(zip(dims[:-1], dims[1:]))
        
        conv_unit = partial(ResnetBlock, groups=resnet_grnorm_groups)
        #self.init_conv = nn.Conv2d(in_planes, init_planes, 1, padding = 0)
        init_conv = [nn.Conv2d(in_planes, init_planes, 1, padding = 0)]
                
        layers = []
        for ind, (dim_in, dim_out) in enumerate(in_out):            
            for i in range(resnet_stacks):
                layers.append(conv_unit(dim_in, dim_in))
            if attention:
                layers.append(Residual(PreNorm(dim_in, LinearAttention(dim_in))))
            layers.append(Downsample(dim_in, dim_out, downsample_mode, pool_kern))

        if last_resnet:
            for i in range(resnet_stacks):
                layers.append(conv_unit(dim_out, dim_out))

        #self.encoder = nn.Sequential(*layers)
        layers = init_conv + layers
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.encoder(x)
        #return self.encoder(self.init_conv(x))
        


