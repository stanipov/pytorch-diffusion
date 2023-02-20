import torch.nn as nn
import torch, math
from inspect import isfunction

#  ------------------------------------------------------- 
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim, embed_param = 10000):
        """
        embed_param - a magical parameter that everyone uses as 10'000
        """
        super().__init__()
        self.dim = dim
        self.T = embed_param

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(self.T) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
#  ------------------------------------------------------- 

def exists(x):
    return x is not None
#  ------------------------------------------------------- 

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
#  ------------------------------------------------------- 

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr
#  ------------------------------------------------------- 

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
#  -------------------------------------------------------        
        
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
#  -------------------------------------------------------                
        
        
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
