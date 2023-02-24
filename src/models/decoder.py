from torch import nn
import torch
from functools import partial

from src.models.helpers import exists, default, PreNorm, Residual
from src.models.conv_blocks import ResnetBlock, Upsample
from src.models.attention import LinearAttention

class Decoder(nn.Module):
    def __init__(self, in_planes = 512,
                 out_planes = 3,
                 plains_divs = [8, 4, 2, 1], 
                 resnet_grnorm_groups = 4,
                 resnet_stacks = 2):
        super().__init__()
           
        init_planes = in_planes // max(plains_divs)
        dims = [init_planes, *map(lambda m: init_planes * m, plains_divs[::-1])] 
        in_out = list(zip(dims[:-1], dims[1:]))[::-1]
       
        conv_unit = partial(ResnetBlock, groups=resnet_grnorm_groups)
                    
        layers = []
        for ind, (dim_out, dim_in) in enumerate(in_out):            
            for i in range(resnet_stacks):
                layers.append(conv_unit(dim_in, dim_in))
            layers.append(Residual(PreNorm(dim_in, LinearAttention(dim_in))))
            layers.append(Upsample(dim_in, dim_out))
            
        self.post_dec = nn.Conv2d(dim_out, out_planes, 1, padding = 0)
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.post_dec(self.encoder(x))
