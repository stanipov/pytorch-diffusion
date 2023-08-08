#!/ext4/pyenv/diffusers/bin/python
#!/home/sf/data/linux/pyenv/pt2/bin/python
#
import pickle

import torch
from tqdm import tqdm
import time, os, json
import numpy as np
from pprint import pprint
from shutil import copy2

from src.train.util import *
from src.datasets.huggan import set_dataloader_unet_hf as set_dataloader_unet
from src.datasets.huggan import LUT
from src.utils.aux import unscale_tensor, save_grid_imgs, get_num_params
from src.models.unet import set_unet
from src.models.diffusion import Diffusion
from src.train.util import set_ema_model

# to avoid IO errors with massive reading of the files
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def sample_rnd_lbls(size, classes):
    """ Returns a tensor of (size, len(classes)) of random integers """
    rnd = []
    for num_items in classes:
        rnd.append(torch.randint(low=0, high=num_items-1, size = (size,)))
    return torch.stack(rnd).float().T

def main(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    msg = f"""
    ==============================================================
                    Training of latent diffusion
                    Training on HF WikiArt using 
                    artist, style, and genre as 
                    conditions for generation
                         Running on {device}
    ==============================================================   
    """
    print(msg)
    print('\nThis config will be used\n')
    pprint(config)

    with open(config_file, 'r') as f:
        config = json.load(f)
    print('\nThis config will be used\n')
    pprint(config)

    # ----------------------------------------------
    # Preconfig
    # ----------------------------------------------
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'

    seed = 19880122
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.preferred_linalg_library(backend='default')
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision('high')

    # model name
    model_name = config['unet_config']['name']

    # results and chkpts folders:
    cwd = config['results']['root']
    if cwd == 'cwd':
        cwd = os.getcwd()
    results_folder = os.path.join(cwd, model_name)
    chkpts = os.path.join(cwd, *(model_name, 'chkpts'))
    img_folder = os.path.join(cwd, *(model_name, 'samples'))
    os.makedirs(chkpts, exist_ok=True)
    os.makedirs(img_folder, exist_ok=True)

    # copy the config
    _f_dst = os.path.join(results_folder, 'train_config.json')
    copy2(config_file, _f_dst)

    # Dataloader
    train_loader, classes = set_dataloader_unet(config)
    num_classes = len(classes)

    # mixed precision
    if 'bfloat' in config['training']['fp16'] or 'bf16' in config['training']['fp16']:
        fp16 = torch.bfloat16
    if config['training']['fp16'] == 'fp16':
        fp16 = torch.float16
    if not config['training']['fp16']:
        fp16 = None

    # Train parameters
    num_epochs = int(config['training']['num_epochs'])
    grad_step_acc = int(config['training']['grad_step_acc'])
    sample_every = int(config['training']['sample_every'])
    save_every = int(config['training']['save_every'])
    save_snapshot = config['training']['save_snapshot']
    flag_compile = config['training']['compile']
    if save_snapshot:
        print(f'Model snapshots will be saved in {chkpts}')

    # Set single GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device for model: ", torch.cuda.get_device_name(device))

    # Set the UNet
    print('Setting up the UNet')
    unet_self_cond = config['unet_config'].get('self_condition', False)
    config['unet_config']['classes'] = classes
    unet_raw = set_unet(config['unet_config'])
    if config['training']['compile']:
        unet = torch.compile(unet_raw).to(device)
    else:
        unet = unet_raw.to(device)
    print(f'\tParameters: {get_num_params(unet):,}')

    # EMA UNet
    ema_flag = False
    if 'ema' in config:
        ema_flag = config['ema'].get('enabled', False)
    if ema_flag:
        ema_unet = set_ema_model(unet_raw, config['ema'])
    else:
        ema_unet = None

    # Set the AutoEncoder model
    if 'vq' in config['autoencoder']['type']:
        autoencoder, autoencoder_eager = prepare_vqmodel(config, device, flag_compile, 'autoencoder')
        encoder_tanh = config['autoencoder'].get('encode_tanh_out', False)
        vq_model = True
    elif 'vae' in config['autoencoder']['type']:
        autoencoder, autoencoder_eager = prepare_vaemodel(config, device, flag_compile, config_key = 'autoencoder')
        vq_model = False

    print('Setting optimizer, scheduler, and scaler')
    # set the model optimizers
    optimizer = set_optimizer(config['optimizer'], unet, flag_compile)

    # Schedulers
    scheduler = None
    if 'type' not in config['lr_scheduler']:
        raise ValueError
    if config['lr_scheduler']['type'].lower() == 'onecyclelr':
        config['lr_scheduler']['epochs'] = num_epochs
        config['lr_scheduler']['steps_per_epoch'] = int(
            np.ceil(len(train_loader) / config['training']['grad_step_acc']))
    scheduler = set_lr_scheduler(config['lr_scheduler'], optimizer)

    if fp16:
        init_scale = config['training'].get('init_scale', 16384)
        growth_interval = config['training'].get('growth_interval', 10)
        growth_factor = config['training'].get('growth_factor', 2)
        backoff_factor = config['training'].get('backoff_factor', 0.5)
        scaler = torch.cuda.amp.GradScaler(init_scale=init_scale,
                                             growth_factor=growth_factor,
                                             growth_interval=growth_interval,
                                             backoff_factor=backoff_factor)
    else:
        scaler = None

    # Diffusion
    print('Setting the DDPM class')
    sampling_batch = config['sampling'].get('sampling_batch', 4)
    grid_rows = config['sampling'].get('grid_rows', 2)
    eta = config['sampling'].get('eta', 1.0)
    noise_dict = config['noise']
    timesteps = config['diffusion']['timesteps']
    ddim_skip = config['diffusion']['skip']
    loss_type = config['diffusion']['loss']
    diffusion = Diffusion(noise_dict, unet, timesteps,
                     loss=loss_type,
                     sample_every=ddim_skip,
                     device=device)

    if ema_unet:
        ema_diffusion = Diffusion(noise_dict, ema_unet.ema_model, timesteps,
                         loss=loss_type,
                         sample_every=ddim_skip,
                         device=device)
    else:
        ema_diffusion = None

    # ---------------------------------------------------
    # Loading optimizer and scaler from a chkpt if exists
    # ---------------------------------------------------
    start_epoch = 0
    step = 0
    chkpt_path = config['training'].get('chkpt', None)
    if chkpt_path:
        print(f'Loading opt., scaler, sch. from {chkpt_path}')
        checkpoint = torch.load(chkpt_path)
        if 'scheduler' in checkpoint:
            msg = scheduler.load_state_dict(checkpoint['scheduler'])
            print(msg)
            _lr = scheduler.get_lr()
            print(f'\tWill start with lr={_lr}')
        if 'scaler' in checkpoint:
            if scaler:
                msg = scaler.load_state_dict(checkpoint['scaler'])
                print(msg)
                print(f'\tScaler: {scaler.get_scale()}')
        if 'optimizer' in checkpoint:
            msg = optimizer.load_state_dict(checkpoint['optimizer'])
            print(msg)
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
            print(f'\tStarting from epoch: {start_epoch+1}')
        if step in checkpoint:
            step = checkpoint['step']
            print(f'\tGlobal step: {start_epoch + 1}')
        print('Done')

    if save_snapshot:
        print(f'Model snapshots will be saved in {chkpts}')
    print(f'Training for {num_epochs-start_epoch} epochs')
    print(f'Sampling every {sample_every} steps')
    print(f'Saving every {save_every} steps')
    print('==============================================================\n\t\tTraining\n==============================================================\n')
    losses = []

    autoencoder.eval()
    m_loss_fname = os.path.join(results_folder, 'model_log_loss.csv')
    train_start = time.time()
    for epoch in range(start_epoch, num_epochs):
        t_start = time.time()
        progress_bar = tqdm(train_loader, desc=f'Train {epoch+1}', total = len(train_loader),
                            mininterval = 1.0, leave=False, disable=False, colour = '#009966',
                            dynamic_ncols=True)
        
        epoch_avg_loss = 0
        optimizer.zero_grad(set_to_none = True)
        for bstep, batch in enumerate(progress_bar):
            # get x, labels, and encode the x
            x = batch[0].to(device)
            x_lbls = batch[1].float().to(device)

            with torch.cuda.amp.autocast(dtype=fp16, cache_enabled=False) and torch.no_grad():
                if vq_model:
                    enc_x = autoencoder.encode(x, encoder_tanh)*autoencoder.scaling_factor
                else:
                    enc_x = autoencoder.encode(x)*autoencoder.scaling_factor
    
            # calculate the loss
            t = torch.randint(0, timesteps, (x.shape[0],), device=device).long()
            with torch.cuda.amp.autocast(dtype=fp16):
                loss = diffusion.get_loss(enc_x, t, noise=None, x_self_cond=unet_self_cond, classes=x_lbls)
                loss = loss / grad_step_acc
            
            # Accumulates scaled (not scaled) gradients
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # update scaler, optimizer, and backpropagate
            if step != 0 and step % grad_step_acc == 0  or (bstep+1) == len(train_loader):
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none = True)
                else:                   
                    optimizer.step()
                    optimizer.zero_grad(set_to_none = True)
                # update EMA model
                if ema_unet:
                    ema_unet.update()

            if scheduler:
                lr = scheduler.get_last_lr()[0]
            else:
                lr = 0
            losses.append([loss.item(), lr])
            epoch_avg_loss += losses[-1][0]
            
            msg_dict = {
                f'Step {step} loss': f'{losses[-1][0]:.5f}',
            }    
            progress_bar.set_postfix(msg_dict)
                        
            # save generated images and checkpoints
            if step != 0 and (step+1) % sample_every == 0:
                print(f'\n\tSampling at epoch {epoch+1} and step {step+1}')
                sample_lbls = sample_rnd_lbls(sampling_batch, classes)
                print(f'\t\t\samle lbls: {sample_lbls.shape}')
                sample_lbls = sample_lbls.to(device)
                sampling_size = (sampling_batch, enc_x.shape[1], enc_x.shape[2], enc_x.shape[3])
                # sample
                print('\t\tNon-EMA diffusion')
                t_sample_start = time.time()
                with torch.cuda.amp.autocast(dtype=fp16, cache_enabled=False) and torch.no_grad():
                    samples = diffusion.p_sample(sampling_size,
                                                 x_self_cond=unet_self_cond,
                                                 classes=sample_lbls,
                                                 last=True, eta=eta)
                print(f'\tSampled in {(time.time() - t_sample_start):.2f} sec.')
                # decode
                with torch.cuda.amp.autocast(dtype=fp16, cache_enabled=False) and torch.no_grad():
                    Y = autoencoder.decode(samples.to(device)/autoencoder.scaling_factor)
                # unscale, make the grid, and save
                all_images = unscale_tensor(Y)
                save_grid_imgs(all_images, max(1, all_images.shape[0] // grid_rows), f'{img_folder}/sample-s_{step+1}-e_{epoch+1}.jpg')
                print(f'\nSaved at\n\t{img_folder}/sample-s_{step+1}-e_{epoch+1}.jpg\n')
                torch.save(unet_raw.state_dict(), f'{chkpts}/sample_snapshot_unet.pt')
                if ema_unet:
                    print('\t\tNon-EMA diffusion')
                    t_sample_start = time.time()
                    with torch.cuda.amp.autocast(dtype=fp16, cache_enabled=False) and torch.no_grad():
                        samples = ema_diffusion.p_sample(sampling_size,
                                                     x_self_cond=unet_self_cond,
                                                     classes=sample_lbls,
                                                     last=True, eta=eta)
                    print(f'\tSampled in {(time.time() - t_sample_start):.2f} sec.')
                    # decode
                    with torch.cuda.amp.autocast(dtype=fp16, cache_enabled=False) and torch.no_grad():
                        Y = autoencoder.decode(samples.to(device) / autoencoder.scaling_factor)
                    # unscale, make the grid, and save
                    all_images = unscale_tensor(Y)
                    save_grid_imgs(all_images, max(1, all_images.shape[0] // grid_rows),
                                   f'{img_folder}/EMA_sample-s_{step + 1}-e_{epoch + 1}.jpg')
                    with open(f'{chkpts}/sample_snapshot_FULL-EMA_unet.pkl', 'wb') as f:
                        pickle.dump(ema_unet, f)
                    torch.save(ema_unet.ema_model.state_dict(), f'{chkpts}/sample_snapshot_EMA_unet.pt')
                
            if step != 0 and (step+1) % save_every == 0:
                torch.save(unet_raw.state_dict(), f'{chkpts}/snapshot_unet.pt')
                checkpoint = {
                    'classes': classes,
                    'epoch': epoch,
                    'step': step,
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict() if scaler else None,
                    'scheduler': scheduler.state_dict() if scheduler else None
                }
                torch.save(checkpoint, f'{chkpts}/train_chkpt_s{step+1}-e{epoch+1}.pt')

            step += 1
                
        if scheduler:
            try:
                scheduler.step()
            except Exception as E:
                print(E)
        
        if save_snapshot:
            torch.save(unet_raw.state_dict(), f'{chkpts}/snapshot_unet.pt')
            if ema_unet:
                with open(f'{chkpts}/snapshot_EMA_unet.pkl', 'wb') as f:
                    pickle.dump(ema_unet, f)
                torch.save(ema_unet.ema_model.state_dict(), f'{chkpts}/snapshot_EMA_unet.pt')

            checkpoint = {
                    'epoch': epoch,
                    'step': step,
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict() if scaler else None,
                    'scheduler': scheduler.state_dict() if scheduler else None
                }
            torch.save(checkpoint, f'{chkpts}/train_snapshot.pt')
        #np.savetxt(f'{chkpts}/loss.dat', np.array(losses))
        
        epoch_avg_loss /= (bstep+1)
        msg = f'\t----> Mean loss: {epoch_avg_loss:<3.5f}'
        print(f'\t----> Epoch {epoch+1} in {time.time() - t_start:.2f}')
        t = np.array(losses)
        np.savetxt(m_loss_fname, t, header='loss, lr', delimiter=",")
        print(msg)


    torch.save(unet_raw.state_dict(), f'{chkpts}/FINAL_model_e{epoch+1}.pt')
    print(f'Saved the final model at\n\t{chkpts}/FINAL_model_e{epoch+1}.pt')

    if ema_unet:
        with open(f'{chkpts}/FINAL_EMA-model_e{epoch+1}.pkl', 'wb') as f:
            pickle.dump(ema_unet, f)
        torch.save(ema_unet.ema_model.state_dict(), f'{chkpts}/FINAL-EMA_unet_e{epoch+1}.pt')
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler else None,
        'scheduler': scheduler.state_dict() if scheduler else None
    }
    torch.save(checkpoint, f'{chkpts}/FINAL_training_e{epoch+1}.pt')
    print(f'Saved the final states of scheduler, optimizer, scaler at\n\t{chkpts}/FINAL_training_e{epoch+1}.pt')
    dt = time.time() - train_start
    if dt>= 3600:
        dt /= 3600
        print(f'Finished training in {dt} hrs.')
    else:
        dt /= 60
        print(f'Finished training in {dt} min.')
    print(f'Tra')
    return 0
                
        
# ===================================================================    
if __name__ == '__main__':
    import argparse
    
    arg_desc = '''\
        Training of Vq-VAE image compressor for latent diffusion
        -------------------------------------------------------
                Please, provide name to the config file
        '''
    
    parser = argparse.ArgumentParser(
                            formatter_class = argparse.RawDescriptionHelpFormatter,
                            description= arg_desc)
                            
    parser.add_argument("-cfg", metavar="config file", help = "Path to the config", required=True)
    args = parser.parse_args()
    cfg_file = args.cfg
    
    main(cfg_file)
