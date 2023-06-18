from torch import nn
import torch
from functools import partial

from src.models.helpers import exists, default, PreNorm, Residual
from src.models.conv_blocks import ResnetBlock, Upsample, WeightStandardizedConv2d
from src.models.attention import LinearAttention

# ================================================================================================
def scale_tensor_11(tensor):
    tmin = tensor.min().item()
    tmax = tensor.max().item() 
    return (tensor - tmin)/(abs(tmax-tmin)) * 2 -1
    
    
class Decoder2(nn.Module):
    def __init__(self, 
                 in_planes = 4,
                 init_planes = 64,
                 out_planes = 3,
                 plains_divs = (8, 4, 2, 1), 
                 resnet_grnorm_groups = 4,
                 resnet_stacks = 2,
                 #last_resnet = False,
                 up_mode = 'bilinear',
                 scale = 2,
                 attention = [],
                 attn_heads = None,
                 attn_dim = None,
                 eps = 1e-6,
                 legacy_mid = False,
                 tanh_out = False,
                 legacy_out = False
                 ):
        super().__init__()
        
        if not attn_heads:
            attn_heads = 4
        if not attn_dim:
            attn_dim = 32
           
        self.tanh = tanh_out
        
        dims = [init_planes, *map(lambda m: init_planes * m, plains_divs[::-1])] 
        in_out = list(zip(dims[1:], dims[:-1]))[::-1]
        
        #print(f'\t\tDecoder: init_planes {init_planes}')
        #print(f'\t\tDecoder: plains_mults {plains_divs}')
        #print(f'\t\tDecoder: {in_out}')
       
        conv_unit = partial(ResnetBlock, groups=resnet_grnorm_groups) 

        _layer = [] 
        for ind, (dim_in, dim_out) in enumerate(in_out):            
            is_last = ind == len(in_out) - 1
            
            #print(f'Layer {ind}: in: {dim_in}, out: {dim_out}')
            
            for i in range(resnet_stacks):
                _layer.append(conv_unit(dim_in, dim_in))
            if dim_out in attention or ind in attention:
                #print(f'\t\tDecoder: Using attention in {ind} with dim_in {dim_in}, {attn_heads} heads of {attn_dim} dim')
                _layer.append(Residual(PreNorm(dim_in, LinearAttention(dim_in, attn_heads, attn_dim))))
            if is_last:
                _up = WeightStandardizedConv2d(in_channels=dim_in, 
                                               out_channels=dim_out, 
                                               kernel_size=3, padding=1)
            else:
                _up = Upsample(dim_in, dim_out, up_mode, scale)
            _layer.append(_up)
        self.upscale = nn.Sequential(*_layer)
        
        
        self.conv_in = WeightStandardizedConv2d(in_channels=in_planes, 
                                                out_channels=in_out[0][0], 
                                                kernel_size=3, padding=1)
        

        # mid block (mirroring such of the encoder)
        
        _in_planes = in_out[0][0]
        _layer = []        
        if legacy_mid:
            for i in range(resnet_stacks):
                _layer.append(conv_unit(_in_planes, _in_planes))
        else:
            _layer.append(conv_unit(_in_planes, _in_planes))
            _layer.append(Residual(PreNorm(_in_planes, LinearAttention(_in_planes, attn_heads, attn_dim) )) )
            _layer.append(conv_unit(_in_planes, _in_planes))
        self.mid_block = nn.Sequential(*_layer)

        # post decoder
        if legacy_out:
            self.post_up = nn.Sequential(
                            nn.GroupNorm(num_groups=resnet_grnorm_groups,
                                         num_channels=dim_out,
                                         eps = eps),
                            nn.SiLU(),
                            WeightStandardizedConv2d(in_channels=in_out[-1][1], 
                                                   out_channels=out_planes, 
                                                   kernel_size=3, padding=1))
        else:
            self.post_up = nn.Sequential(
                            nn.GroupNorm(num_groups=resnet_grnorm_groups,
                                         num_channels=dim_out,
                                         eps = eps),
                            WeightStandardizedConv2d(in_channels=in_out[-1][1], 
                                                   out_channels=out_planes, 
                                                   kernel_size=3, padding=1))
                
    def forward(self, x):
        x = self.conv_in(x)
        x = self.mid_block(x)
        x = self.upscale(x)
        x = self.post_up(x)
        if self.tanh:
            x = torch.tanh(x)
        return x
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
