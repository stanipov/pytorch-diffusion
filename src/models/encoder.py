from torch import nn
import torch
from functools import partial

from src.models.helpers import exists, default, PreNorm, Residual
from src.models.conv_blocks import ResnetBlock, Downsample
from src.models.attention import LinearAttention

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
