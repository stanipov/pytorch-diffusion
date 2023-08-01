import torch.nn as nn

from src.models.decoder import Decoder2
from src.models.encoder import Encoder2
from src.models.conv_blocks import WeightStandardizedConv2d
from src.models.distributions import DiagonalGaussianDistribution

from typing import Tuple, Optional, List

# ------------------------------------------------- VAE model ----------------------------------------------------------
class VAE(nn.Module):
    def __init__(self,
                 img_ch: int = 3,
                 enc_init_ch: int = 64,
                 ch_mult: Tuple[int] = (1, 2, 4, 4),
                 grnorm_groups: int = 8,
                 resnet_stacks: int = 2,
                 latent_dim: int = 4,
                 embed_dim = None,
                 down_mode: str = 'avg',
                 down_kern: int = 2,
                 down_attn: Optional[List[int]] = [],
                 up_mode: str = 'bilinear',
                 up_scale: int = 2,
                 up_attn: Optional[List[int]] = [],
                 attn_heads: Optional[int] = 4,
                 attn_dim: Optional[int] = 8,
                 eps: Optional[float] = 1e-6,
                 scaling_factor: Optional[float] = 0.18215,
                 dec_tanh_out: bool = False):

        super().__init__()
        embed_dim = latent_dim if embed_dim is None else embed_dim

        self._encoder = Encoder2(in_planes = img_ch,
                                 init_planes = enc_init_ch,
                                 plains_mults = ch_mult,
                                 resnet_grnorm_groups = grnorm_groups,
                                 resnet_stacks = resnet_stacks,
                                 downsample_mode = down_mode,
                                 pool_kern = down_kern,
                                 attention = down_attn,
                                 latent_dim = latent_dim *2,
                                 eps = eps,
                                 legacy_mid = False,
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
                                 legacy_mid = False,
                                 attn_heads = attn_heads,
                                 attn_dim = attn_dim,
                                 tanh_out = dec_tanh_out)

        self.pre_quantizer = WeightStandardizedConv2d(latent_dim*2, 2*embed_dim, kernel_size=1)
        self.post_quantizer = WeightStandardizedConv2d(embed_dim, latent_dim, kernel_size=1)
        self.scaling_factor = scaling_factor

    def _encode(self, x):
        h = self._encoder(x)
        moments = self.pre_quantizer(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def encode(self, x):
        h = self._encoder(x)
        moments = self.pre_quantizer(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior.sample()

    def decode(self, z):
        z = self.post_quantizer(z)
        dec = self._decoder(z)
        return dec

    def forward(self, x, sample_posterior=True):
        posterior = self._encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_last_layer(self):
        return self._decoder.post_up[1].weight