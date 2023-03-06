#!/home/sf/data/linux/pyenv/pt1/bin/python
import torch
from torch import optim

from src.models.vq_vae import VQ_VAE, set_codec
from src.models.unet import Unet, set_unet
from src.models.ddpm import Diffusion

from src.datasets.artbench import artbench_hires, artbench256
from torch.utils.data import Subset
from src.utils.aux import unscale_tensor, save_grid_imgs, get_num_params, cos_schedule

from tqdm import tqdm
import time, os, json
import numpy as np
from pprint import pprint
from shutil import copy2

from src.models.noise_schedulers import cosine_beta_schedule
from src.models.helpers import *

# to avoid IO errors with massive reading of the files
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def main(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    msg = f"""
    ==============================================================
                    Training of latent diffusion
     on ArtBench dataset (https://github.com/liaopeiyuan/artbench)
                         Running on {device}
    ==============================================================   
    """
    print(msg)
    
    
    print('\nThis config will be used\n')
    pprint(config) 
    
    
    # TORCH_CUDNN_V8_API_ENABLED=1 
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.preferred_linalg_library(backend='default')
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    
    # model name
    model_name = config['unet_config']['model_name']
        
    # results and chkpts folders:
    cwd = config['results']['root']
    if cwd == 'cwd':
        cwd = os.getcwd()
    results_folder = os.path.join(cwd, *('results', model_name))
    chkpts         = os.path.join(cwd, *('results', model_name, 'chkpts'))
    img_folder     = os.path.join(cwd, *('results', model_name, 'samples'))
    os.makedirs(chkpts, exist_ok = True)
    os.makedirs(img_folder, exist_ok = True)
    
    # copy the config
    _f_dst = os.path.join(results_folder, 'train_config.json')
    copy2(config_file, _f_dst)
        
    # Dataset params
    image_size = config['dataset']['image_size']
    root = config['dataset']['location']
    use_subset = config['dataset']['use_subset']
    if use_subset:
        use_subset = float(use_subset)
    img_resize = config['dataset']['img_resize']
        
    # training params
    if  'bfloat' in config['training']['fp16']:
        fp16 = torch.bfloat16
    if config['training']['fp16'] == 'fp16':
        fp16 = torch.float16
    if not config['training']['fp16']:
        fp16 = None
    num_epochs         = int(config['training']['num_epochs'])
    batch_size         = int(config['training']['batch_size'])
    dataloader_workers = int(config['training']['dataloader_workers'])
    sample_every       = int(config['training']['sample_every'])
    save_every         = int(config['training']['save_every'])
    save_snapshot      = config['training']['save_snapshot']
    grad_step_acc      = config['training']['grad_step_acc']
    grad_step_acc      = grad_step_acc if grad_step_acc else 1
    
    # optimizer
    lr    = float(config['optimizer']['lr'])
    betas = config['optimizer']['betas']
    eps   = float(config['optimizer']['eps'])
    
    # lr_scheduler
    lr_scheduler_enabled = config['lr_scheduler']['enabled']
    milestones           = config['lr_scheduler']['milestones']
    gamma                = float(config['lr_scheduler']['gamma'])
    
    # Codec
    codec_train_config = config['codec']['train_config']
    codec_w            = config['codec']['weights']
    
    # Unet
    unet_cfg = config['unet_config']
    unet_w   = unet_cfg['load']
    unet_w = None if not unet_w else unet_w
    if unet_w:
        print(f'Loading UNet weights from\n\t{unet_w}')
    
    # Diffusion
    timesteps = config['diffusion']['timesteps']
    if config['diffusion']['noise_scheduler'] == 'cosine':
        noise_scheduler = cosine_beta_schedule
    else:
        raise("Not implemented")
    
    print('Setting the dataset')
    if img_resize and image_size > 256:
        dataset = artbench_hires(root, image_size=image_size)
    else:
        dataset = artbench256(root)
    num_classes = len(dataset.classes)
    
    if use_subset:
        print(f'\t{use_subset} of the total dataset will be used')
        subset_idx = np.random.choice(len(dataset), size = int(len(dataset)*use_subset), replace = False).astype(int)
        np.savetxt(f'{results_folder}/tain_idxs.dat', subset_idx)
        dataset = Subset(dataset, subset_idx) #range(0,int(len(dataset)*use_subset))
        print(f'\t{len(dataset)} of images will be used')
    else:
        print(f'Using whole of {len(dataset)} images')
    print(f'{num_classes} classes were found')

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=dataloader_workers,
                                              pin_memory = True)
    
    
    print('Loading the codec')
    codec = set_codec(codec_train_config, codec_w)
    encoder = codec._encoder
    encoder.eval()
    encoder = encoder.to(device)
    VQ = codec._vq
    VQ.eval()
    decoder = codec._decoder
    decoder.eval()
    print(f'Encoder is moved to {device}\nVQ and decoder reside in CPU\nDone')
    
    print(f'\tParameters: {get_num_params(codec):,}')
    print('Done')
    
    print('Setting up the UNet')
    unet = set_unet(unet_cfg, unet_w)
    unet = unet.to(device)
    print(f'\tParameters: {get_num_params(unet):,}')
    print('Done')
    
    print('Setting the optimizer, lr schedler, and the scaler')
    optimizer = optim.AdamW(params = unet.parameters(),lr = lr, betas=betas, eps = eps,  )

    scheduler = None
    if lr_scheduler_enabled:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                           milestones=milestones, 
                                           gamma=gamma, last_epoch=-1, verbose=True)
    if fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    print('Setting the DDPM class')
    ddpm = Diffusion(noise_scheduler, unet, timesteps)
        
    
    # reread config    
    with open(config_file, 'r') as f:
        config = json.load(f)
    unet_out_ch = config['unet_config']['out_channels']
    num_classes = config['unet_config']['num_classes']
        
    if save_snapshot:
        print(f'Model snapshots will be saved in {chkpts}')
    print(f'Training for {num_epochs} epochs')
    print(f'Sampling every {sample_every} steps')
    print(f'Saving every {save_every} steps')
    print('==============================================================\n\t\tTraining\n==============================================================\n')
    losses = []
        
    step = 0
    for epoch in range(num_epochs):
        t_start = time.time()
        progress_bar = tqdm(train_loader, desc=f'Train {epoch+1}', total = len(train_loader), leave=False, disable=False)
        
        epoch_avg_loss = 0
        optimizer.zero_grad(set_to_none = True)
        for bstep, batch in enumerate(progress_bar):
            # get x, labels, and encode the x
            x = batch[0].to(device)
            x_lbls = batch[1].to(device)
            t = torch.randint(0, timesteps, (x.shape[0],), device=device).long()
            with torch.cuda.amp.autocast(dtype=fp16, cache_enabled=False) and torch.no_grad():
                enc_x = encoder(x)
    
            # calculate the loss
            loss = ddpm.loss(enc_x, t, loss_type="huber", fp16=fp16, classes = x_lbls) 
            loss = loss / grad_step_acc
            
            # Accumulates scaled (not scaled) gradients
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # update scaler, optimizer, and backpropagate
            if step != 0 and (step+1) % grad_step_acc == 0  or (step+1) == len(train_loader):
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none = True)
                else:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none = True)
                
            losses.append(loss.item())
            epoch_avg_loss += losses[-1]
            
            msg_dict = {
                f'Step {step} loss': f'{losses[-1]:.5f}',
            }    
            progress_bar.set_postfix(msg_dict)
                        
            # save generated images and checkpoints
            if step != 0 and (step+1) % sample_every == 0:
                sample_size =  4
                sample_lbls = torch.randint(low = 0, high = 10, size = (sample_size, )).to(device)
                samples = ddpm.sample(image_size = x.shape[-1], 
                                      batch_size = sample_size, 
                                      channels = unet_out_ch,
                                      classes = sample_lbls, fp16 = fp16)
                with torch.cuda.amp.autocast(dtype=fp16, cache_enabled=False) and torch.no_grad():
                    _, z_q, *n = VQ(samples[-1].to('cpu'))
                    Y = decoder(z_q)
                all_images = unscale_tensor(Y)
                save_grid_imgs(all_images, max(1, all_images.shape[0] // 2), f'{img_folder}/sample-s_{step+1}-e_{epoch+1}.jpg')
                
            if step != 0 and (step+1) % save_every == 0:
                checkpoint = {
                    'model': unet.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict()
                }
                torch.save(checkpoint, f'{chkpts}/chkpt_{step+1}-{epoch+1}.pt')
                #torch.save(unet.state_dict(), f'{chkpts}/model_{step+1}-{epoch+1}.pt')
                
            step += 1
                
        if scheduler:
            scheduler.step()  
        
        if save_snapshot:
            torch.save(unet.state_dict(), f'{chkpts}/model_snapshot.pt')
            checkpoint = {
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'scheduler': scheduler.state_dict() 
                }
            torch.save(checkpoint, f'{chkpts}/train_snapshot_s{step+1}-e{epoch+1}.pt')
        np.savetxt(f'{chkpts}/loss.dat', np.array(losses))
        
        epoch_avg_loss /= (bstep+1)
        msg = f'\t----> Mean loss: {epoch_avg_loss:<3.5f}'
        print(f'\t----> Epoch {epoch+1} in {time.time() - t_start:.2f}')
        print(msg)    
    
    torch.save(unet.state_dict(), f'{chkpts}/FINAL_model_{epoch+1}.pt')
    print(f'Saved the final modet at\n\t{chkpts}/FINAL_model_{epoch+1}.pt')
    return 0
                
        
# ===================================================================    
if __name__ == '__main__':
    import argparse
    
    arg_desc = '''\
        Training of Vq-VAE image compressor for latent diffusion
        on ArtBench dataset (https://github.com/liaopeiyuan/artbench)
        -------------------------------------------------------
                Plese, provide name to the config file
        '''
    
    parser = argparse.ArgumentParser(
                            formatter_class = argparse.RawDescriptionHelpFormatter,
                            description= arg_desc)
                            
    parser.add_argument("-cfg", metavar="config file", help = "Path to the config", required=True)
    args = parser.parse_args()
    cfg_file = args.cfg
    
    main(cfg_file)
