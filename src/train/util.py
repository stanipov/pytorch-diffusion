from tqdm import tqdm
import torch
from torch import optim
import torch.nn.functional as F
from src.models.vq_vae import VQModel
from src.models.vae import VAE
from src.utils.aux import get_num_params
from src.losses.discriminator import NLayerDiscriminator, weights_init
from src.losses.lpips import init_lpips_loss
from src.models.ema import EMA


def conv2tqdm_msg(msg, fmt = '>.5f'):
    res = {}
    for key in msg:
        if key != 'Step':
            res[key] = f'{msg[key]:{fmt}}'
        else:
            res[key] = f'{msg[key]}'
    return res


def partial_load_model(model, saved_model_path):
    """ https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/31 """
    pretrained_dict = torch.load(saved_model_path, map_location='cpu')
    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                       (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    status = model.load_state_dict(pretrained_dict, strict=False)
    print(status)

    return model


def save_model(model, fname):
    torch.save(model.state_dict(), fname)

def set_optimizer(config, model, flag_compile = False):
    # optimizer: model
    lr                 = float(config['lr'])
    betas              = config['betas']
    eps                = float(config['eps'])
    weight_decay       = float(config.get('weight_decay', 1e-3))
    return optim.AdamW(params = model.parameters(),
                       lr = lr, betas=betas, eps = eps, 
                       weight_decay = weight_decay,
                       fused = True)


def set_lr_scheduler(config, optimizer):
    if config['type'].lower() == 'onecyclelr':
        print('Setting OneCycleLR')
        return set_lr_scheduler_OneCycleLR(config, optimizer)
    if config['type'].lower() == 'multisteplr':
        print('Setting MultiStepLR')
        return set_lr_scheduler_MultiStepLR(config, optimizer)


def set_lr_scheduler_OneCycleLR(config, optimizer):
    max_lr = config['max_lr']
    total_steps=config.get('total_steps', None)
    epochs=config.get('epochs', None)
    steps_per_epoch=config.get('steps_per_epoch', None)
    pct_start=config.get('pct_start', 0.3)
    anneal_strategy=config.get('anneal_strategy', 'cos')
    cycle_momentum=config.get('cycle_momentum', True)
    base_momentum=config.get('base_momentum', [0.4, 0.80])
    max_momentum=config.get('max_momentum', [0.55, 0.91])
    div_factor=config.get('div_factor', 25.0)
    final_div_factor=config.get('final_div_factor', 10000.0)
    three_phase=config.get('three_phase', False)
    print(f'\tepochs {epochs}')
    print(f'\tsteps_per_epoch {steps_per_epoch}')
    print(f'\ttotal_steps {total_steps}')
    return optim.lr_scheduler.OneCycleLR(optimizer, max_lr = max_lr,
                                        total_steps = total_steps,
                                        epochs = epochs,
                                        steps_per_epoch = steps_per_epoch,
                                        pct_start = pct_start,
                                        anneal_strategy = anneal_strategy,
                                        cycle_momentum = cycle_momentum,
                                        base_momentum = base_momentum,
                                        max_momentum = max_momentum,
                                        div_factor = div_factor,
                                        final_div_factor = final_div_factor,
                                        three_phase = three_phase,
                                        verbose=False)

    
def set_lr_scheduler_MultiStepLR(config, optimizer):
    lr_scheduler_enabled = config['enabled']
    milestones           = config['milestones']
    gamma                = float(config['gamma'])
    scheduler = None
    if lr_scheduler_enabled:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                           milestones=milestones, 
                                           gamma=gamma,
                                           last_epoch=-1, verbose=False)
    return scheduler


def set_ema_model(unet_model, cfg):
    beta = cfg.get('beta', 0.9)
    update_after_step = cfg.get('update_after_step', 10)
    update_every = cfg.get('update_every', 10)
    inv_gamma = cfg.get('inv_gamma', 1.0)
    power = cfg.get('power', 0.9)
    return EMA(model=unet_model,
                   beta=0.9,
                   update_after_step=10,
                   update_every=10,
                   inv_gamma=1.0,
                   power=0.9)



# -------------------------------------------------- Set up models -----------------------------------------------------
def set_VQModel(config, load=False, config_key=None):
    """ Sets the VQ Model from a config  """
    # Model params
    if config_key is None:
        if 'model' in config:
            cfg = config['model']
        elif 'vqmodel' in config:
            cfg = config['vqmodel']
        else:
            raise KeyError("Can't find model config!")
    else:
        cfg = config[config_key]

    img_ch = cfg.get('img_channels', 3)
    enc_init_ch = cfg.get('enc_init_channels', 64)
    ch_mult = cfg.get('ch_mult', (1, 2, 4, 4))
    grnorm_groups = cfg.get('grnorm_groups', 4)
    resnet_stacks = cfg.get('resnet_stacks', 2)
    latent_dim = cfg.get('latent_dim', 4)
    num_vq_embeddings = cfg.get('num_vq_embeddings', 256)
    vq_embed_dim = cfg.get('vq_embed_dim', None)
    commitment_cost = cfg.get('commitment_cost', 0.25)
    down_mode = cfg.get('down_mode', 'max')
    down_kern = cfg.get('down_kern', 2)
    down_attn = cfg.get('down_attn', [])
    up_mode = cfg.get('up_mode', 'nearest')
    up_scale = cfg.get('up_scale', 2)
    up_attn = cfg.get('up_attn', [])
    eps = cfg.get('eps', 1e-6)
    scaling_factor = cfg.get('scaling_factor', 0.18215)
    attn_heads = cfg.get('attn_heads', None)
    attn_dim = cfg.get('attn_dim', None)
    legacy_mid = cfg.get('legacy_mid', False)
    dec_tanh_out = cfg.get('dec_tanh_out', False)

    _model = VQModel(
        img_ch=img_ch,
        enc_init_ch=enc_init_ch,
        ch_mult=ch_mult,
        grnorm_groups=grnorm_groups,
        resnet_stacks=resnet_stacks,
        latent_dim=latent_dim,
        num_vq_embeddings=num_vq_embeddings,
        vq_embed_dim=vq_embed_dim,
        commitment_cost=commitment_cost,
        down_mode=down_mode,
        down_kern=down_kern,
        down_attn=down_attn,
        up_mode=up_mode,
        up_scale=up_scale,
        up_attn=up_attn,
        eps=eps,
        scaling_factor=scaling_factor,
        attn_heads=attn_heads,
        attn_dim=attn_dim,
        legacy_mid=legacy_mid,
        dec_tanh_out=dec_tanh_out
    )  # .apply(weights_init)

    if load:
        print(f'\tLoading the pretrained weights from\n\t{load}')
        try:
            status = _model.load_state_dict(torch.load(load), strict=True)
            print(f'\t{status}')
        except Exception as E:
            print(E)
            _model = partial_load_model(_model, load)

    return _model


def set_VAEModel(config, load=False, config_key=None):
    """ Sets the VAE Model from a config  """
    # Model params
    if config_key is None:
        if 'model' in config:
            cfg = config['model']
        elif 'vaemodel' in config:
            cfg = config['vaemodel']
        elif 'vae' in config:
            cfg = config['vae']
        elif 'vae_model' in config:
            cfg = config['vae_model']
        else:
            raise KeyError("Can't find model config!")
    else:
        cfg = config[config_key]

    img_ch = cfg.get('img_channels', 3)
    enc_init_ch = cfg.get('enc_init_channels', 64)
    ch_mult = cfg.get('ch_mult', (1, 2, 4, 4))
    grnorm_groups = cfg.get('grnorm_groups', 4)
    resnet_stacks = cfg.get('resnet_stacks', 2)
    latent_dim = cfg.get('latent_dim', 4)
    embed_dim = cfg.get('embed_dim', None)
    down_mode = cfg.get('down_mode', 'max')
    down_kern = cfg.get('down_kern', 2)
    down_attn = cfg.get('down_attn', [])
    up_mode = cfg.get('up_mode', 'nearest')
    up_scale = cfg.get('up_scale', 2)
    up_attn = cfg.get('up_attn', [])
    eps = cfg.get('eps', 1e-6)
    scaling_factor = cfg.get('scaling_factor', 0.18215)
    attn_heads = cfg.get('attn_heads', None)
    attn_dim = cfg.get('attn_dim', None)
    dec_tanh_out = cfg.get('dec_tanh_out', False)

    _model = VAE(
        img_ch=img_ch,
        enc_init_ch=enc_init_ch,
        ch_mult=ch_mult,
        grnorm_groups=grnorm_groups,
        resnet_stacks=resnet_stacks,
        latent_dim=latent_dim,
        embed_dim=embed_dim,
        down_mode=down_mode,
        down_kern=down_kern,
        down_attn=down_attn,
        up_mode=up_mode,
        up_scale=up_scale,
        up_attn=up_attn,
        eps=eps,
        scaling_factor=scaling_factor,
        attn_heads=attn_heads,
        attn_dim=attn_dim,
        dec_tanh_out=dec_tanh_out
    )  # .apply(weights_init)

    if load:
        print(f'\tLoading the pretrained weights from\n\t{load}')
        try:
            status = _model.load_state_dict(torch.load(load), strict=True)
            print(f'\t{status}')
        except Exception as E:
            print(E)
            _model = partial_load_model(_model, load)

    return _model


def prepare_vqmodel(config, device, flag_compile=False, config_key=None):
    if config_key is None:
        if 'model' in config:
            cfg = config['model']
        elif 'vqmodel' in config:
            cfg = config['vqmodel']
        else:
            raise KeyError("Can't find model config!")
    else:
        cfg = config[config_key]

    load_name       = cfg['load_name']
    print(f'{device} will be used')
    print('Setting the model')
    model_eager = set_VQModel(config, load=load_name, config_key=config_key)
    print('Done')
    print(f'Model parameters: {get_num_params(model_eager):,}')
    if flag_compile:
        print(f'Compiling model')
        model_eager = model_eager.to(device)
        model = torch.compile(model_eager) 
        print('Done')
    else:
        model = model_eager.to(device)
    return model, model_eager


def prepare_vaemodel(config, device, flag_compile=False, config_key=None):
    if config_key is None:
        if 'model' in config:
            cfg = config['model']
        elif 'vaemodel' in config:
            cfg = config['vaemodel']
        elif 'vae' in config:
            cfg = config['vae']
        elif 'vae_model' in config:
            cfg = config['vae_model']
        else:
            raise KeyError("Can't find model config!")
    else:
        cfg = config[config_key]

    load_name       = cfg['load_name']
    print(f'{device} will be used')
    print('Setting the model')
    model_eager = set_VAEModel(config, load_name, config_key)
    print('Done')
    print(f'Model parameters: {get_num_params(model_eager):,}')
    if flag_compile:
        print(f'Compiling model')
        model_eager = model_eager.to(device)
        model = torch.compile(model_eager)
        print('Done')
    else:
        model = model_eager.to(device)
    return model, model_eager

# ----------------------------------------------- Set up Discriminator -------------------------------------------------
def init_discriminator(config, device):
    print('\n------------------------\nSetting discriminator')
    disc_in_channels = config.get('disc_in_channels', 3)
    disc_num_layers = config.get('disc_num_layers', 3)
    disc_ndf = config.get('disc_ndf', 64)
    use_actnorm = config.get('use_actnorm', False)
    load = config.get('load_name', '')

    model = NLayerDiscriminator(input_nc=disc_in_channels, ndf=disc_ndf,
                                n_layers=disc_num_layers,
                                use_actnorm=use_actnorm).apply(weights_init)
    print(f'Disc parameters: {get_num_params(model):,}')
    if load:
        print(f'Loading pretrained weights from:\n\t{load}')
        # status = model.load_state_dict(torch.load(load), strict = False)
        status = partial_load_model(model, load)
        print(f'\t{status}')
    print('Done\n------------------------')
    return model.to(device)

# -------------------------------------------------- Set up losses -----------------------------------------------------
def set_lpips_loss(config, device, flag_compile = False):
    print(f'Setting the loss')
    lpips_cfg = config['lpips']
    lpips_loss_eager = init_lpips_loss(lpips_cfg)
    if flag_compile:
        lpips_loss_eager = lpips_loss_eager.to(device)
        lpips_loss = torch.compile(lpips_loss_eager) 
    else:
        lpips_loss = lpips_loss_eager.to(device)
    print('Done')  
    return lpips_loss

# -------------------------------------------------- Pretrain disc -----------------------------------------------------
def pretrain_discriminator(disc, vqmodel, dataloader, loss_fn, d_opt, 
                           d_sch, d_scaler, epochs, device, fp16, grad_step_acc):
    step = 0 
    for epoch in range(epochs):

        avg_tot = 0
        progress_bar = tqdm(dataloader, desc=f'Train {epoch+1}', total = len(dataloader),
                                    mininterval = 1.0, leave=False, disable=False, colour = '#009966')
        
        avg_metrics = {}
        for bstep, X in enumerate(progress_bar):
            d_opt.zero_grad(set_to_none = True)
            batch_size = X[0].shape[0]
            batch = X[0].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype = fp16):
                with torch.no_grad():
                    vq_recon, *_ = vqmodel(batch)
                d_loss, msg = loss_fn(batch, vq_recon)

            # Accumulates scaled (not scaled) gradients
            if d_scaler:
                d_scaler.scale(d_loss).backward()
            else:
                d_loss.backward()

            # update scaler, optimizer, and backpropagate
            if step != 0 and step % grad_step_acc == 0  or (bstep+1) == len(dataloader):
                if d_scaler:
                    d_scaler.step(d_opt)
                    d_scaler.update()
                    d_opt.zero_grad(set_to_none = True)
                else:                   
                    d_opt.step()
                    d_opt.zero_grad(set_to_none = True)

            m_msg_tqdm = conv2tqdm_msg(msg)
            m_msg_tqdm[f'Step'] = step
            progress_bar.set_postfix(m_msg_tqdm)
            
            # add to the avg metrics
            for key in msg:
                if key not in avg_metrics:
                    avg_metrics[key] = 0
                avg_metrics[key] += msg[key]
            
            step += 1
            
        msg = '\t---->'
        for key in avg_metrics:
            if key != 'Step':
                msg += f' {key}: {avg_metrics[key]/(bstep + 1):>.5f};'
        print(msg)
        
    return disc
