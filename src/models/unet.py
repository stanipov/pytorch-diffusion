from torch import nn, einsum
import torch
from functools import partial

from src.models.helpers import exists, default, PreNorm, SinusoidalPositionEmbeddings, Residual
from src.models.conv_blocks import ResnetBlock, Downsample, Upsample
from src.models.attention import Attention, LinearAttention

from src.train.util import partial_load_model

class Unet(nn.Module):
    def __init__(
        self,
        img_size,
        init_dim = None,
        dim_mults = (1, 2, 4, 8),
        time_dim = None,
        in_channels = 3,
        out_channels = 3,
        down_mode = 'avg',
        down_kern = 2,
        up_mode = 'nearest',
        up_scale = 2,
        resnet_stacks = 2,
        attn_heads = 4,
        attn_head_res = 32,
        self_condition = False,
        resnet_grnorm_groups = 4,
        classes = None
    ):
        """
        img_size             - assuming square image of (img_size, img_size)
        init_dim             - out planes from the pre-Unet layer (i.e. in_channels -> init_dim)
        dim_mults            - tuple of multipliers to multiply element-wise init_dim,
        time_dim             - dimensinality of time embeddings, if None: 4*img_size
        in_channels          - number of input channels
        out_channels         - number of output channels of the UNet
        self_condition       - legace from Hugging Face implenetation. IDK
        resnet_grnorm_groups - numbers for group norm for each  ResNet blocks
        resnet_stacks        - number of ResNet blocks
        classes              - number of classes or list of integers for number of classes for each label/type
        down_mode:           - if "conv", then strided convolution is used, 
                             "avg" ot "max" - average2D or maxPool2D are used respectively 
        down_kern:           - Size of the pooling kernel, has effect only if down_mode is avg or max
        up_mode:             - If 'conv", strided transposed convolution is used, otherwise, interpolation
        up_scale             - Upscale interpolation factor, default 2

        Adopted from: https://huggingface.co/blog/annotated-diffusion with modifications:
        - module lists init
        - replaced mid-attention with linear attention
        - added class conditions
        - replaced 'manual' down and up with a for-loop
        """
        super().__init__()

        # determine dimensions
        if not out_channels:
            out_channels = in_channels
        self.self_condition = self_condition
        input_channels = in_channels * (2 if self_condition else 1)
        init_dim = init_dim if init_dim else img_size
        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)] 
        in_out = list(zip(dims[:-1], dims[1:]))
        self.resnet_stacks = resnet_stacks

        # time embeddings
        if not time_dim:
            time_dim = img_size * 4
        
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding = 0)        
        conv_unit = partial(ResnetBlock, groups=resnet_grnorm_groups)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(img_size),
            nn.Linear(img_size, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # Down
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList([conv_unit(dim_in, dim_in, time_emb_dim=time_dim) for _ in range(self.resnet_stacks)])
                +
                nn.ModuleList(
                    [
                        Residual(PreNorm(dim_in, LinearAttention(dim = dim_in, heads=attn_heads, dim_head=attn_head_res))),
                        Downsample(dim_in, dim_out, down_mode, down_kern)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = conv_unit(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(dim=mid_dim, heads=attn_heads, dim_head=attn_head_res))) # replaced with Linear attention
        #self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(dim = mid_dim, heads=attn_heads, dim_head=attn_head_res)))
        self.mid_block2 = conv_unit(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Up
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList([conv_unit(dim_out + dim_in, dim_out, time_emb_dim=time_dim) for _ in range(self.resnet_stacks)])
                +
                nn.ModuleList(
                    [
                        Residual(PreNorm(dim_out, LinearAttention(dim=dim_out, heads=attn_heads, dim_head=attn_head_res))),
                        Upsample(dim_out, dim_in, up_mode, up_scale)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.final_res_block = conv_unit(dim_in*2, img_size, time_emb_dim=time_dim) # dim * 2
        self.final_conv = nn.Conv2d(img_size, out_channels, 1)
        
        if classes:
            if type(classes) == int:
                self.classes = [classes]
            else:
                self.classes = classes
            # nn.Embeddings are commented out for now
            # the new approach with MLP should be flexible
            # when multi-class labels are supplied
            #self.lbl_embeds = nn.Embedding(classes, time_dim)
            self.lbl_embeds = nn.Sequential(*[
                nn.Linear(len(self.classes), time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            ])

    def forward(self, x, time, x_self_cond=None, lbls = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)
        
        if lbls is not None:
            t += self.lbl_embeds(lbls)

        h = []

        # downwards descent
        for i, layer in enumerate(self.downs):
            # iterate over resnet blocks
            for block in layer[0:self.resnet_stacks]:
                x = block(x,t)
                h.append(x)
            # attention
            x = layer[self.resnet_stacks](x)
            # downsampler
            x = layer[-1](x)
        
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)     
                
        # upwards ascent
        for i, layer in enumerate(self.ups):
            # iterate over resnet blocks
            for bi, block in enumerate(layer[0:self.resnet_stacks]):
                x_down = h.pop()
                x = torch.cat((x,  x_down), dim=1)
                x = block(x,t)
            # attention
            x = layer[self.resnet_stacks](x)
            # upsampler
            x = layer[-1](x)

        x = torch.cat((x, r), dim=1)
      
        x = self.final_res_block(x, t)
        return self.final_conv(x)
# ----------------------------------------------------------------------------------------
    
def set_unet(config_dict):
    img_size             = config_dict['img_size']
    init_dim             = config_dict.pop('init_dim', None)   
    dim_mults            = config_dict.pop('dim_mults', (1, 2, 4, 8))
    time_dim             = config_dict.pop('time_dim', None)
    in_channels          = config_dict.pop('in_channels', 3)
    out_channels         = config_dict.pop('out_channels', 3)
    self_condition       = config_dict.pop('self_condition', False)
    resnet_grnorm_groups = config_dict.pop('resnet_grnorm_groups', 4)
    resnet_stacks        = config_dict.pop('resnet_stacks', 2)
    classes              = config_dict.pop('classes', None)
    down_mode            = config_dict.pop('classes', 'avg')
    down_kern            = config_dict.pop('down_kern', 2)
    up_mode              = config_dict.pop('up_mode', 'bilinear')
    up_scale             = config_dict.pop('up_scale', 2)
    attn_heads           = config_dict.pop('attn_heads', 4)
    attn_head_res        = config_dict.pop('attn_head_res', 16)

    model = Unet(
        img_size             = img_size,
        init_dim             = init_dim,
        dim_mults            = dim_mults,
        time_dim             = time_dim,
        in_channels          = in_channels,
        out_channels         = out_channels,
        self_condition       = self_condition,
        resnet_grnorm_groups = resnet_grnorm_groups,
        resnet_stacks        = resnet_stacks, 
        classes              = classes,
        down_mode            = down_mode, 
        down_kern            = down_kern,
        up_mode              = up_mode,
        up_scale             = up_scale,
        attn_heads           = attn_heads,
        attn_head_res        = attn_head_res
    )

    if 'load_name' in config_dict:
        weights = config_dict['load_name']
    else:
        weights = None

    if weights:
        print(f'Loading model weights from\n\t{weights}')
        model = partial_load_model(model, weights)
        
    return model
