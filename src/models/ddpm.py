import torch, time
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, Tuple
from src.models.noise_schedulers import cosine_beta_schedule, linear_beta_schedule, quadratic_beta_schedule, sigmoid_beta_schedule
from src.losses.util import loss_fn

# ------------------------------------------ Newer version ------------------------------------------


def get_betas(timesteps: int, noise_kw: Dict):
    """
    Computes betas according to the scheduler type and required parameters.
    The function has default parameters.
    SF
    """
    scheduler_name = noise_kw['type']
    if 'cosine' in  scheduler_name:
        scheduler = cosine_beta_schedule
        s = noise_kw.get('s', 0.008)
        betas = scheduler(timesteps, s)
    elif 'linear' in scheduler_name:
        scheduler = linear_beta_schedule
        beta_start = noise_kw.get('beta_start', 0.0001)
        beta_end = noise_kw.get('beta_end', 0.02)
        betas = scheduler(timesteps, beta_start, beta_end)
    elif 'quadratic' in scheduler_name:
        scheduler = quadratic_beta_schedule
        beta_start = noise_kw.get('beta_start', 0.0001)
        beta_end = noise_kw.get('beta_end', 0.02)
        betas = scheduler(timesteps, beta_start, beta_end)
    elif 'sigmoid' in scheduler_name:
        scheduler = sigmoid_beta_schedule
        beta_start = noise_kw.get('beta_start', 0.0001)
        beta_end = noise_kw.get('beta_end', 0.02)
        betas = scheduler(timesteps, beta_start, beta_end)
    else:
        raise ValueError(f'Scheduler type of "{scheduler_name}" is not recognized')
    return betas

def ddpm_schedules(beta_t: float, T: int) -> Dict[str, torch.Tensor]:
    """
    Returns pre-computed schedules for DDPM sampling, training process.
    Adopted from https://github.com/cloneofsimo/minDiffusion/blob/master/mindiffusion/ddpm.py
    """
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,  # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
    }

class DDPM(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        noise_kw: Dict,
        n_T: int,
        loss_func: str = 'huber'
    ) -> None:
        # Adopted from https://github.com/cloneofsimo/minDiffusion/blob/master/mindiffusion/ddpm.py
        super().__init__()
        self.eps_model = eps_model

        betas = get_betas(n_T, noise_kw)

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        for k, v in ddpm_schedules(betas, n_T).items():
            self.register_buffer(k, v)

        self.n_T = n_T
        self.criterion = loss_fn(loss_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Makes forward diffusion x_t, and tries to guess epsilon value from x_t using eps_model.
        This implements Algorithm 1 in the paper.
        """
        _ts = torch.randint(1, self.n_T + 1, (x.shape[0],)).to(x.device)
        # t ~ Uniform(0, n_T)
        eps = torch.randn_like(x)  # eps ~ N(0, 1)

        x_t = (
            self.sqrtab[_ts, None, None, None] * x
            + self.sqrtmab[_ts, None, None, None] * eps
        )  # This is the x_t, which is sqrt(alphabar) x_0 + sqrt(1-alphabar) * eps
        # We should predict the "error term" from this x_t. Loss is what we return.

        return self.criterion(eps, self.eps_model(x_t, _ts / self.n_T))

    def sample(self, n_sample: int, size, device) -> torch.Tensor:

        x_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        # This samples accordingly to Algorithm 2. It is exactly the same logic.
        for i in range(self.n_T, 0, -1):
            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0
            eps = self.eps_model(
                x_i, torch.tensor(i / self.n_T).to(device).repeat(n_sample, 1)
            )
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )

        return x_i

# ------------------------------------------ Older version from HF tutorial ------------------------------------------
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
#  -------------------------------------------------------

def forward_diffusion_params(noise_scheduler, timesteps):
    betas = noise_scheduler(timesteps=timesteps)
    # define alphas 
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    return (betas, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance)
#  -------------------------------------------------------

class Diffusion:
    def __init__(self, noise_scheduler, model, timesteps = 100, sample_every = 10):
        
        self.noise_scheduler = noise_scheduler
        self.model = model
        self.timesteps = timesteps
        dif_params = forward_diffusion_params(self.noise_scheduler, timesteps)
        self.betas = dif_params[0]
        self.sqrt_recip_alphas = dif_params[1]
        self.sqrt_alphas_cumprod = dif_params[2]
        self.sqrt_one_minus_alphas_cumprod = dif_params[3]
        self.posterior_variance = dif_params[4]
        self.sample_every = sample_every

    def q_sample(self, x_start, t, noise = None):
        """
        Prior sample
        """
        if noise is None:
            noise = torch.randn_like(x_start)
                  
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)      
                     
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def loss(self, x_start, t, noise=None, x_self_cond = None, classes = None,
             loss_type="l1", fp16 = torch.float16):
        
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_noisy = self.q_sample(x_start = x_start, 
                                t = t, noise = noise)
        
        if fp16:
            amp_enabled = True
        else:
            amp_enabled = False
        
        with torch.cuda.amp.autocast(dtype = fp16, enabled = amp_enabled):
                predicted_noise = self.model(x_noisy, t, x_self_cond = x_self_cond, lbls = classes)
                       
        #print(f'x_start        : {x_start.shape}')   
        #print(f'noise          : {noise.shape}')        
        #print(f'predicted_noise: {predicted_noise.shape}')        

        with torch.cuda.amp.autocast(dtype = fp16, enabled = amp_enabled):        
            if loss_type == 'l1':
                loss = F.l1_loss(noise, predicted_noise)
            
            elif loss_type == 'l2':   
                loss = F.mse_loss(noise, predicted_noise)
            
            elif loss_type == "huber":
                loss = F.smooth_l1_loss(noise, predicted_noise)
            
            else:
                raise NotImplementedError()
        
        return loss
    
    def _p_sample(self, x, t, t_index, fp16 = torch.float16, x_self_cond = None, classes = None):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod,
                                                  t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
    
        if fp16:
            amp_enabled = True
        else:
            amp_enabled = False
            
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype = fp16, enabled = amp_enabled):
                model_out = self.model(x, t, x_self_cond = x_self_cond, lbls = classes)
                model_mean = sqrt_recip_alphas_t * (
                        x - betas_t * model_out / sqrt_one_minus_alphas_cumprod_t)
                
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise 
        
        
    def _p_sample_loop(self, shape, fp16 = torch.float16, x_self_cond = None, classes = None, return_all_steps = False):
        t_now = time.time()
        with torch.no_grad():
            device = next(self.model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []
        
        progress_bar = tqdm(reversed(range(0, self.timesteps)), desc=f'Sampling every {self.sample_every}', 
                            total = self.timesteps,
                            mininterval = 1.0,  
                            colour = '#FFCC00')

        #for i in tqdm(reversed(range(0, self.timesteps)), desc=f'Sampling every {self.sample_every}', total=self.timesteps):
        for i in progress_bar:
            if i % self.sample_every == 0 or i == 0:
#                print(f'\n{i}\n')
                img = self._p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), 
                                     i, fp16, x_self_cond, classes)
            if return_all_steps:                 
                    imgs.append(img.cpu())
        msg_dict = {
                f'Sampled in': f'{time.time()-t_now:.2f} seconds',
            }    
        progress_bar.set_postfix(msg_dict)
        if return_all_steps:
            return imgs
        else:
            return img

    def sample(self, image_size, batch_size=16, channels=3, x_self_cond = None, classes = None, fp16 = torch.float16):
        return self._p_sample_loop(shape=(batch_size, channels, image_size, image_size), 
                                   fp16 = fp16, x_self_cond = x_self_cond, classes = classes)
