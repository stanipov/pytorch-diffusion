#!/home/sf/data/linux/pyenv/pt2/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.datasets.artbench import set_dataloader_vq
from src.models.vq_vae import VQModel, set_VQModel
from src.losses.lpips import LPIPS_VQ_loss, set_loss
from src.utils.aux import unscale_tensor, save_grid_imgs, get_num_params, cos_schedule
import torch.nn.functional as F

import torch
from torch import optim

from PIL import Image
import torchvision
from torchvision import transforms
from torch.utils.data import Subset

from tqdm import tqdm
import time, os, json
import numpy as np
from pprint import pprint
from shutil import copy2

# to avoid IO errors with massive reading of the files
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# ===================================================================    
def main(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print('\nThis config will be used\n')
    pprint(config)           

    # TORCH_CUDNN_V8_API_ENABLED=1 
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.preferred_linalg_library(backend='default')
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision('high')

    #torch._dynamo.config.verbose=True

    # model name
    model_name = config['model']['name']

    # results and chkpts folders:
    cwd = config['results']['root']
    if cwd == 'cwd':
        cwd = os.getcwd()
    results_folder = os.path.join(cwd, model_name)
    chkpts         = os.path.join(cwd, *(model_name, 'chkpts'))
    img_folder     = os.path.join(cwd, *(model_name, 'samples'))
    os.makedirs(chkpts, exist_ok=True)
    os.makedirs(img_folder, exist_ok = True)

    # copy the config
    _f_dst = os.path.join(results_folder, 'train_config.json')
    copy2(config_file, _f_dst)
    
    # Dataloader
    train_loader = set_dataloader_vq(config)

    # Model params   
    load_name       = config['model']['load_name']
    load            = config['model']['load']

    # Train parameters
    num_epochs         = int(config['training']['num_epochs'])
    grad_step_acc      = int(config['training']['grad_step_acc'])
    batch_size         = int(config['training']['batch_size'])
    dataloader_workers = int(config['training']['dataloader_workers'])
    sample_every       = int(config['training']['sample_every'])
    save_every         = int(config['training']['save_every'])
    save_snapshot      = config['training']['save_snapshot']
    flag_compile       = config['training']['compile']
    if save_snapshot:
        print(f'Model snapshots will be saved in {chkpts}')
        
    if  'bfloat' in config['training']['fp16'] or 'bf16' in config['training']['fp16']:
        fp16 = torch.bfloat16
    if config['training']['fp16'] == 'fp16':
        fp16 = torch.float16
    if not config['training']['fp16']:
        fp16 = None

    # optimizer
    lr    = float(config['optimizer']['lr'])
    betas = config['optimizer']['betas']
    eps   = float(config['optimizer']['eps'])

    # lr_scheduler
    lr_scheduler_enabled = config['lr_scheduler']['enabled']
    milestones           = config['lr_scheduler']['milestones']
    gamma                = float(config['lr_scheduler']['gamma'])

    # set the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'{device} will be used')
    print('Setting the model')
    model_eager = set_VQModel(config, load_name)
    print('Done')
    print(f'Model parameters: {get_num_params(model_eager):,}')
    if flag_compile:
        print(f'Compiling model')
        model_eager = model_eager.to(device)
        model = torch.compile(model_eager) 
        print('Done')
    else:
        model = model_eager.to(device)
    
    total_loss   = []
    step = 0

    x_original = next(iter(train_loader))
    x_original = x_original[0]
    save_grid_imgs(unscale_tensor(x_original), max(x_original.shape[0] // 4, 2), f'{img_folder}/original-images.jpg')

    print(f'Training for {num_epochs} epochs')
    print(f'Sampling every {sample_every} steps')
    print(f'Saving every {save_every} steps')
    print('==============================================================\n\t\tTraining\n==============================================================\n')
    for epoch in range(num_epochs):
        t_start = time.time()
        progress_bar = tqdm(train_loader, desc=f'Train {epoch+1}', total = len(train_loader),
                            mininterval = 1.0, leave=False, disable=False, colour = '#009966')

        avg_tot = 0
        avg_recon = 0
        avg_quant = 0
        avg_percep = 0

        for bstep, X in enumerate(progress_bar):
            optimizer.zero_grad()
            batch_size = X[0].shape[0]
            batch = X[0].to(device, non_blocking=False)

            with torch.cuda.amp.autocast(dtype = fp16):
                batch_recon, q_loss, perplexity_s, _, _ = model(batch)
                loss, msg = lpips_loss(batch, batch_recon, q_loss)
             
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

            total_loss.append(loss.item())      
            progress_bar.set_postfix(msg)
            np.savetxt(f'{chkpts}/total_loss.dat', np.array(total_loss))
            
            # save generated images and checkpoints
            if step != 0 and (step+1) % sample_every == 0:
                with torch.no_grad():
                    x_recon, _, _, _, _ = model(x_original.to(device))
                x_recon = unscale_tensor(x_recon)
                save_grid_imgs(x_recon, max(x_recon.shape[0] // 4, 2), f'{img_folder}/recon_imgs-{step+1}-{epoch+1}.jpg')

            if step != 0 and (step+1) % save_every == 0:
                checkpoint = {
                    'model_orig': model_eager.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict()
                }
                torch.save(checkpoint, f'{chkpts}/chkpt_{step+1}-{epoch+1}.pt')
                torch.save(model.state_dict(), f'{chkpts}/model_compiled_{step+1}-{epoch+1}.pt')
                torch.save(model_eager.state_dict(), f'{chkpts}/model_orig_{step+1}-{epoch+1}.pt')
            step += 1


        if save_snapshot:
            torch.save(model.state_dict(), f'{chkpts}/model_compiled_snapshot.pt')
            torch.save(model_eager.state_dict(), f'{chkpts}/model_orig_snapshot.pt')

        if scheduler:
            scheduler.step()   

        avg_tot /= (bstep+1)
        
        msg = f'\t----> loss: {avg_tot:<2.5f}'
        print(f'Epoch {epoch+1} in {time.time() - t_start:.2f}')
        print(msg)
    return 0
        
# ===================================================================    
if __name__ == '__main__':
    import argparse
    
    arg_desc = '''\
        Training of VQ-AE image compressor for latent diffusion
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
    
    msg = """
    ==============================================================
        Training of VQ-AE image compressor for latent diffusion
     on ArtBench dataset (https://github.com/liaopeiyuan/artbench)
    ==============================================================   
    """
    print(msg)
    main(cfg_file)
