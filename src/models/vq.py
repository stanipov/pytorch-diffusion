import torch
import torch.nn as nn
import torch.nn.functional as F
#import numpy as np
#from torch import einsum
from einops import rearrange
#  -------------------------------------------------------
"""
The code below is adopted from https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb 
and https://huggingface.co/blog/annotated-diffusion
and implements an algorithm presented in 'Neural Discrete Representation Learning' by van den Oord et al.
https://arxiv.org/abs/1711.00937

These 2 VQ implementations use channels as the space to which quantize (num_embeddings) 
with 'embedding_dim'-dimensional vectors. In the application of interest to VQ-VAE, 
num_embeddings is the number of planes from VAE's encoder, e.g. 512. embedding_dim is then 
dimensionality of each vector in the latent space, e.g. 256.

"""
# -----------------------------------------------------------------------------------------
class VectorQuantizer(nn.Module):
    """
    Credit to 
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py
    
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, unknown_index="random",
                 sane_index_shape=False):
        super().__init__()
        self.n_e = num_embeddings
        self.e_dim = embedding_dim
        self.beta = commitment_cost

        self._embedding = nn.Embedding(self.n_e, self.e_dim)
        #self._embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        rng = max(1e-3, 1.0 / self.n_e)
        self._embedding.weight.data.uniform_(-1*rng, rng)

        self.sane_index_shape = sane_index_shape


    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        # convert inputs from BCHW -> BHWC
        #z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self._embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self._embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self._embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])
        # loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings, encoding_indices
        return loss, z_q, perplexity, min_encodings, min_encoding_indices

    def get_codebook_entry(self, indices, shape):
        # get quantized latent vectors
        z_q = self._embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q
# -----------------------------------------------------------------------------------------

class old_VectorQuantizer(nn.Module):
    """
    Respectfully borrowed from 
    https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        # outputs: loss, quantized vector q_z, perplexity, encodings, and respective indeces
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings, encoding_indices
    
    def sample(self, encoding_indices, out_size, device = 'cuda'):
        """
        From a given encoding indeces produce a quantized output 
        of "out_size"
        """
        encoding_indices=encoding_indices.view(-1,1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).to(device) 
        encodings.scatter_(1, encoding_indices, 1)
        z_q = torch.matmul(encodings, self._embedding.weight).view(-1, 
                                                                   out_size[0], 
                                                                   out_size[1], 
                                                                   self._num_embeddings)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q
# -----------------------------------------------------------------------------------------

class VectorQuantizerEMA(nn.Module):
    """Sonnet module representing the VQ-VAE layer.
    https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    
    Respectfully borrowed from 
        https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb

    Implements the algorithm presented in
    'Neural Discrete Representation Learning' by van den Oord et al.
    https://arxiv.org/abs/1711.00937

    Input any tensor to be quantized. Last dimension will be used as space in
    which to quantize. All other dimensions will be flattened and will be seen
    as different examples to quantize.

    The output tensor will have the same shape as the input.
    For example a tensor with shape [16, 32, 32, 64] will be reshaped into
    [16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
    independently.

    Attributes:
    embedding_dim: integer representing the dimensionality of the tensors in the
      quantized space. With respect to VAE, this is numpber of planes from 
      its encoder, say, 512
      
    num_embeddings: integer, the number of vectors in the quantized space. 
        Eg, 256 vectors
    commitment_cost: scalar which controls the weighting of the loss terms 
    decay - expinential decay, e.g decay = 0.99
      
      
    """
        
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
                
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
