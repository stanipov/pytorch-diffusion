from torch import nn, einsum
import torch
from functools import partial

from src.models.helpers import exists, default, PreNorm, SinusoidalPositionEmbeddings, Residual
from src.models.conv_blocks import ResnetBlock, Downsample, Upsample
from src.models.attention import Attention, LinearAttention


class Unet(nn.Module):
    def __init__(
        self,
        img_size,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        time_dim = None,
        channels = 3,
        self_condition = False,
        resnet_grnorm_groups = 4,
        num_classes = None
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, img_size)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding = 0) # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: init_dim * m, dim_mults)] #dim*m
        in_out = list(zip(dims[:-1], dims[1:]))
        
        conv_unit = partial(ResnetBlock, groups=resnet_grnorm_groups)

        # time embeddings
        if not time_dim:
            time_dim = img_size * 4

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

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        conv_unit(dim_in, dim_in, time_emb_dim=time_dim),
                        conv_unit(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = conv_unit(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = conv_unit(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        conv_unit(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        conv_unit(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = conv_unit(dim_in*2, img_size, time_emb_dim=time_dim) # dim * 2
        self.final_conv = nn.Conv2d(img_size, self.out_dim, 1)
        
        if num_classes:
            self.num_classes = num_classes
            self.lbl_embeds = nn.Embedding(num_classes,time_dim)

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

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)
      
        x = self.final_res_block(x, t)
        return self.final_conv(x)
