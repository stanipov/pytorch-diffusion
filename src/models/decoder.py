from torch import nn
import torch
from functools import partial

from src.models.helpers import exists, default, PreNorm, Residual
from src.models.conv_blocks import ResnetBlock, Upsample, WeightStandardizedConv2d
from src.models.attention import LinearAttention

# ================================================================================================
class Decoder2(nn.Module):
    def __init__(self, 
                 in_planes = 4,
                 init_planes = 64,
                 out_planes = 3,
                 plains_divs = (8, 4, 2, 1), 
                 resnet_grnorm_groups = 4,
                 resnet_stacks = 2,
                 last_resnet = False,
                 up_mode = 'bilinear',
                 scale = 2,
                 attention = False,
                 eps = 1e-6
                 ):
        super().__init__()
           
        dims = [init_planes, *map(lambda m: init_planes * m, plains_divs[::-1])] 
        in_out = list(zip(dims[:-1], dims[1:]))[::-1]
       
        conv_unit = partial(ResnetBlock, groups=resnet_grnorm_groups)
                    
        layers = []
        
        self.conv_in = WeightStandardizedConv2d(in_channels=in_planes, 
                                                out_channels=init_planes*max(plains_divs), 
                                                kernel_size=3, padding=1)
        _layer = []
        for i in range(resnet_stacks):
            _layer.append(conv_unit(init_planes*max(plains_divs), init_planes*max(plains_divs)))
        self.mid_block = nn.Sequential(*_layer)
        
        _layer = [] 
        for ind, (dim_out, dim_in) in enumerate(in_out):            
            is_last = ind == len(in_out) - 1
            for i in range(resnet_stacks):
                _layer.append(conv_unit(dim_in, dim_in))
            if attention:
                _layer.append(Residual(PreNorm(dim_in, LinearAttention(dim_in))))
            if is_last:
                _up = WeightStandardizedConv2d(in_channels=dim_in, 
                                               out_channels=dim_out, 
                                               kernel_size=3, padding=1)
            else:
                _up = Upsample(dim_in, dim_out, up_mode, scale)
            _layer.append(_up)
        self.upscale = nn.Sequential(*_layer)
            
        self.post_up = nn.Sequential(
                        nn.GroupNorm(num_groups=resnet_grnorm_groups,
                                     num_channels=dim_out,
                                     eps = eps),
                        nn.SiLU(),
                        WeightStandardizedConv2d(in_channels=dim_out, 
                                               out_channels=out_planes, 
                                               kernel_size=3, padding=1))
                
    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid_block(x)
        x = self.upscale(x)
        return self.post_up(x)
# ================================================================================================
# legacy code

class Decoder(nn.Module):
    def __init__(self, in_planes = 512,
                 out_planes = 3,
                 plains_divs = [8, 4, 2, 1], 
                 resnet_grnorm_groups = 4,
                 resnet_stacks = 2,
                 last_resnet = False,
                 up_mode = 'bilinear',
                 scale = 2,
                 attention = False
                 ):
        super().__init__()
           
        init_planes = in_planes // max(plains_divs)
        dims = [init_planes, *map(lambda m: init_planes * m, plains_divs[::-1])] 
        in_out = list(zip(dims[:-1], dims[1:]))[::-1]
       
        conv_unit = partial(ResnetBlock, groups=resnet_grnorm_groups)
                    
        layers = []
        
        for i in range(resnet_stacks):
            layers.append(conv_unit(in_out[0][1], in_out[0][1]))
        
        for ind, (dim_out, dim_in) in enumerate(in_out):            
            for i in range(resnet_stacks):
                layers.append(conv_unit(dim_in, dim_in))
            if attention:
                layers.append(Residual(PreNorm(dim_in, LinearAttention(dim_in))))
            layers.append(Upsample(dim_in, dim_out, up_mode, scale))
            
        if last_resnet:
            post_dec_lst = [conv_unit(dim_out, dim_out) for _ in range(resnet_stacks)] \
                            + \
                            [nn.Conv2d(dim_out, out_planes, 1, padding = 0)]
        else:
            post_dec_lst = [nn.Conv2d(dim_out, out_planes, 1, padding = 0)]
        
        layers += post_dec_lst
        self.decoder = nn.Sequential(*layers)
        
        #self.post_dec = nn.Sequential(*post_dec_lst)
        #self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.decoder(x)
        #return self.post_dec(self.encoder(x))
