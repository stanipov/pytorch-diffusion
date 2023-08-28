import torch, time
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict
from src.models.noise_schedulers import cosine_beta_schedule, linear_beta_schedule, quadratic_beta_schedule, \
    sigmoid_beta_schedule
from src.losses.util import loss_fn

# ----------------------------------------------- Helpers -----------------------------------------------
def get_betas(timesteps: int, noise_kw: Dict):
    """
    Computes betas according to the scheduler type and required parameters.
    The function has default parameters.
    SF
    """
    scheduler_name = noise_kw['type']
    if 'cosine' in scheduler_name:
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
    return torch.tensor(betas)

def extract(a, t, x_shape):
    # Taken from Annotated diffusion model by HuggingFace
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_params(betas, timesteps):
    # Adopted from Annotated diffusion model by HuggingFace
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

# # ----------------------------------------------- DDPM and DDIM -----------------------------------------------
# Adopted from https://github.com/ermongroup/ddim/blob/51cb290f83049e5381b09a4cc0389f16a4a02cc9/functions/denoising.py
def compute_alpha(beta, t):
    #print(f'beta: {beta.device}')
    #print(f't: {t.device}')
    beta = torch.cat([torch.zeros(1).to(t.device), beta.to(t.device)], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def generalized_steps(model_args, seq, model, b, eta):
    with torch.no_grad():
        x = model_args[0]
        self_cond = model_args[1]
        clas_lbls = model_args[2]
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]

        progress_bar = tqdm(zip(reversed(seq), reversed(seq_next)),
                            desc=f'DDIM Sampling', total=len(seq),
                            mininterval=0.5, leave=False,
                            disable=False, colour='#F39C12',
                            dynamic_ncols=True)

        #for i, j in zip(reversed(seq), reversed(seq_next)):
        for i,j in progress_bar:
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            et = model(xt, t, x_self_cond=self_cond, lbls=clas_lbls).detach()
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.detach().to('cpu'))

    return xs, x0_preds

def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to(x.device)

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.detach().to('cpu'))
    return xs, x0_preds

# ------------------------------------------ Newer version ------------------------------------------
class Diffusion():
    def __init__(self, noise_dict: Dict, model,
                 timesteps: int = 500, loss:str = 'huber',
                 sample_every: int = 20,
                 device: str = 'cuda'):
        self.timesteps = timesteps
        self.sample_every = sample_every
        self.dev = device
        betas =  get_betas(timesteps=timesteps, noise_kw = noise_dict)
        dif_params = forward_diffusion_params(betas, timesteps)
        self.betas = dif_params[0]
        self.sqrt_recip_alphas = dif_params[1]
        self.sqrt_alphas_cumprod = dif_params[2]
        self.sqrt_one_minus_alphas_cumprod = dif_params[3]
        self.posterior_variance = dif_params[4]
        self.loss = loss_fn(loss)
        self.model = model

    def q_sample(self, x_start, t, noise=None):
        """
        Prior sample
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def forward_diffusion(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start,
                                t=t, noise=noise)
        return x_noisy

    def get_loss(self, x_start, t, noise=None, x_self_cond=None, classes=None):

        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start,
                                t=t, noise=noise)

        predicted_noise = self.model(x_noisy, t, x_self_cond=x_self_cond, lbls=classes)
        return self.loss(noise, predicted_noise)

    def p_sample(self, size, x_self_cond=None,
                 classes=None, last = True,
                 eta: float = 1.0):
        """ Posterior sample """
        x = torch.randn(*size, device=self.dev)
        seq = range(0, self.timesteps, self.sample_every)
        seq = [int(s) for s in list(seq)]
        model_args = (x, x_self_cond, classes)
        xs = generalized_steps(model_args, seq, self.model, self.betas, eta=eta)
        if last:
            return xs[0][-1]
        else:
            return xs


# ------------------------------------------ Legacy version from HF tutorial ------------------------------------------
# TODO: delete
class _Diffusion:
    def __init__(self, noise_scheduler, model, timesteps=100, sample_every=10):

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

    def q_sample(self, x_start, t, noise=None):
        """
        Prior sample
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def loss(self, x_start, t, noise=None, x_self_cond=None, classes=None,
             loss_type="l1", fp16=torch.float16):

        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start,
                                t=t, noise=noise)

        if fp16:
            amp_enabled = True
        else:
            amp_enabled = False

        with torch.cuda.amp.autocast(dtype=fp16, enabled=amp_enabled):
            predicted_noise = self.model(x_noisy, t, x_self_cond=x_self_cond, lbls=classes)

        # print(f'x_start        : {x_start.shape}')
        # print(f'noise          : {noise.shape}')
        # print(f'predicted_noise: {predicted_noise.shape}')

        with torch.cuda.amp.autocast(dtype=fp16, enabled=amp_enabled):
            if loss_type == 'l1':
                loss = F.l1_loss(noise, predicted_noise)

            elif loss_type == 'l2':
                loss = F.mse_loss(noise, predicted_noise)

            elif loss_type == "huber":
                loss = F.smooth_l1_loss(noise, predicted_noise)

            else:
                raise NotImplementedError()

        return loss

    def _p_sample(self, x, t, t_index, fp16=torch.float16, x_self_cond=None, classes=None):
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
            with torch.cuda.amp.autocast(dtype=fp16, enabled=amp_enabled):
                model_out = self.model(x, t, x_self_cond=x_self_cond, lbls=classes)
                model_mean = sqrt_recip_alphas_t * (
                        x - betas_t * model_out / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def _p_sample_loop(self, shape, fp16=torch.float16, x_self_cond=None, classes=None, return_all_steps=False):
        t_now = time.time()
        with torch.no_grad():
            device = next(self.model.parameters()).device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        progress_bar = tqdm(reversed(range(0, self.timesteps)), desc=f'Sampling every {self.sample_every}',
                            total=self.timesteps,
                            mininterval=1.0,
                            colour='#FFCC00')

        # for i in tqdm(reversed(range(0, self.timesteps)), desc=f'Sampling every {self.sample_every}', total=self.timesteps):
        for i in progress_bar:
            if i % self.sample_every == 0 or i == 0:
                img = self._p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                     i, fp16, x_self_cond, classes)
            if return_all_steps:
                imgs.append(img.cpu())
        msg_dict = {
            f'Sampled in': f'{time.time() - t_now:.2f} seconds',
        }
        progress_bar.set_postfix(msg_dict)
        if return_all_steps:
            return imgs
        else:
            return img

    def sample(self, image_size, batch_size=16, channels=3, x_self_cond=None, classes=None, fp16=torch.float16):
        return self._p_sample_loop(shape=(batch_size, channels, image_size, image_size),
                                   fp16=fp16, x_self_cond=x_self_cond, classes=classes)
