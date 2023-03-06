import torch
import torch.nn.functional as F
from tqdm import tqdm

#  -------------------------------------------------------
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
    def __init__(self, noise_scheduler, model, timesteps = 100):
        
        self.noise_scheduler = noise_scheduler
        self.model = model
        self.timesteps = timesteps
        dif_params = forward_diffusion_params(self.noise_scheduler, timesteps)
        self.betas = dif_params[0]
        self.sqrt_recip_alphas = dif_params[1]
        self.sqrt_alphas_cumprod = dif_params[2]
        self.sqrt_one_minus_alphas_cumprod = dif_params[3]
        self.posterior_variance = dif_params[4]

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
        
        
    def _p_sample_loop(self, shape, fp16 = torch.float16, x_self_cond = None, classes = None):
        with torch.no_grad():
            device = next(self.model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            img = self._p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), 
                                 i, fp16, x_self_cond, classes)
            imgs.append(img.cpu())
        return imgs

    def sample(self, image_size, batch_size=16, channels=3, x_self_cond = None, classes = None, fp16 = torch.float16):
        return self._p_sample_loop(shape=(batch_size, channels, image_size, image_size), 
                                   fp16 = fp16, x_self_cond = x_self_cond, classes = classes)
