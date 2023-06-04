#!/home/sf/data/linux/pyenv/pt2/bin/python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from tqdm import tqdm
import time, os, json
import numpy as np
from pprint import pprint
from shutil import copy2

from src.train.util import *
from src.datasets.artbench import set_dataloader_vq, set_dataloader_disc
from src.losses.disc_loss import DiscLoss
from src.losses.discriminator import init_discriminator
from src.losses.lpips import init_lpips_loss
from src.losses.LPIPSWithDisc import LPIPSWithDiscriminator
from src.utils.aux import unscale_tensor, save_grid_imgs, get_num_params

# to avoid IO errors with massive reading of the files
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def conv2tqdm_msg(msg, fmt = '>.5f'):
    res = {}
    for key in msg:
        if key != 'Step':
            res[key] = f'{msg[key]:{fmt}}'
        else:
            res[key] = f'{msg[key]}'
    return res

# ===================================================================    
def main(config_file):
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
    if config['discriminator']['disc_train_batch']:
        disc_loader = set_dataloader_disc(config)
    else:
        disc_loader = train_loader

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

    # mixed precision
    if  'bfloat' in config['training']['fp16'] or 'bf16' in config['training']['fp16']:
        fp16 = torch.bfloat16
    if config['training']['fp16'] == 'fp16':
        fp16 = torch.float16
    if not config['training']['fp16']:
        fp16 = None

    # device
    if torch.cuda.device_count() >= 2:
        device_model = "cuda:0" if torch.cuda.is_available() else "cpu"
        device_loss = "cuda:1" if torch.cuda.is_available() else "cpu"
    else:
        device_model = device_loss = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device for model? ", torch.cuda.get_device_name(device_model))
    print("Device for loss? ", torch.cuda.get_device_name(device_loss))

    # init the model
    model, model_eager = prepare_vqmodel(config, device_model, flag_compile)
    encoder_tanh = config['model'].get('encode_tanh_out', False)
    # init the disc
    if config['discriminator']['enabled']:
        discriminator = init_discriminator(config['discriminator'], device_loss)
    else:
        discriminator = None

    print('Setting optimizer, scheduler, and scaler')
    # set the model optimizers
    m_opt = set_optimizer(config['optimizer'], model, flag_compile)
    if discriminator:
        d_opt = set_optimizer(config['optimizer_d'], discriminator, flag_compile)
    else:
        d_opt = None

    # Schedulers
    if 'type' not in config['lr_scheduler']:
        raise ValueError
    if config['lr_scheduler']['type'].lower() == 'onecyclelr':
        config['lr_scheduler']['epochs'] = num_epochs
        config['lr_scheduler']['steps_per_epoch'] = int(np.ceil(len(train_loader) / config['training']['grad_step_acc']))
    m_sch = set_lr_scheduler(config['lr_scheduler'], m_opt)
    if discriminator:
        if config['lr_scheduler_d']['type'].lower() == 'onecyclelr':
            config['lr_scheduler_d']['epochs'] = num_epochs
            config['lr_scheduler_d']['steps_per_epoch'] = int(np.ceil(len(train_loader) / config['training']['grad_step_acc']))
        d_sch = set_lr_scheduler(config['lr_scheduler_d'], d_opt)
    else:
        d_sch = None

    # Scaler
    if fp16:
        m_scaler = torch.cuda.amp.GradScaler()
        d_scaler = torch.cuda.amp.GradScaler()
    else:
        m_scaler = None
        d_scaler = None
    print('Done')

    # ----------------------------------------------
    # Discriminator pretrain
    # ----------------------------------------------
    if config['discriminator']['disc_pretrain_epochs']:
        disc_loss = DiscLoss(discriminator, 'hinge')
        d_grad_step_acc = config['discriminator']['disc_grad_acc']
        d_epochs = config['discriminator']['disc_pretrain_epochs']
        print(f'---------------------------------------------\nPretraining the discriminator for {d_epochs:>3} epochs\n---------------------------------------------')
        discriminator = pretrain_discriminator(discriminator, model, disc_loader, 
                                               disc_loss, d_opt, d_sch, 
                                               d_scaler, d_epochs, device_loss, fp16,
                                               d_grad_step_acc)
        disc_name = os.path.join(chkpts, 'discriminator.pth')
        save_model(discriminator, disc_name)
        # re-init the optimizer, scaler, and scheduler
        d_opt = set_optimizer(config['optimizer_d'], discriminator, flag_compile)
        d_sch = set_lr_scheduler(config['lr_scheduler_d'], m_opt)
        d_scaler = torch.cuda.amp.GradScaler()
        print(f'Saved pretrained discriminator in\n\t{disc_name}')
        
    # ----------------------------------------------
    # Set LPIPS with Disc
    # ----------------------------------------------
    disc_weight = float(config['discloss'].get('disc_weight', 1.0))
    disc_factor = float(config['discloss'].get('disc_factor', 1.0))
    disc_start = int( config['training'].get('disc_start', 0) )
    disc_loss_fn = config['training'].get('disc_loss_fn', 'hinge')
    LPIPSDiscLoss = LPIPSWithDiscriminator(discriminator=discriminator, 
                                           disc_start=disc_start, 
                                           discriminator_weight = disc_weight,
                                           disc_factor = disc_factor, 
                                           disc_loss = disc_loss_fn,
                                           cfg = config['lpips']).to(device_loss)

    # ----------------------------------------------
    # Save original images
    # ----------------------------------------------
    x_original = next(iter(train_loader))
    x_original = x_original[0]
    save_grid_imgs(unscale_tensor(x_original), max(x_original.shape[0] // 4, 2), f'{results_folder}/original-images.jpg')
    
    # ----------------------------------------------
    # Training
    # ----------------------------------------------
    total_loss   = []
    step = 0
    disc_name = os.path.join(chkpts, 'discriminator_upd.pth')
    
    print(f'Training for {num_epochs} epochs')
    print(f'Sampling every {sample_every} steps')
    print(f'Saving every {save_every} steps')
    print('==============================================================\n\t\tTraining\n==============================================================\n')
    avg_metrics = {}
    m_loss_log = []
    d_loss_log = []
    m_loss_fname = os.path.join(results_folder, 'model_log_loss.csv')
    d_loss_fname = os.path.join(results_folder, 'disc_log_loss.csv')
    for epoch in range(num_epochs):
        t_start = time.time()
        progress_bar = tqdm(train_loader, desc=f'Train {epoch+1}', total = len(train_loader),
                            mininterval = 1.0, leave=False, disable=False, colour = '#009966')

        avg_tot = 0
        avg_metrics = {}
        
        d_opt.zero_grad(set_to_none = True)
        m_opt.zero_grad(set_to_none = True)
        
        for bstep, X in enumerate(progress_bar):
            batch_size = X[0].shape[0]
            batch = X[0].to(device_model, non_blocking=False)

            # GAN-like part
            with torch.cuda.amp.autocast(dtype = fp16):
                batch_recon, q_loss, perplexity_s, _, _ = model(batch, encoder_tanh)
            with torch.cuda.amp.autocast(dtype = fp16):
                m_loss, m_msg = LPIPSDiscLoss(q_loss.to(device_loss), batch.to(device_loss),
                                              batch_recon.to(device_loss), 0, step,
                                              last_layer = model.get_last_layer().to(device_loss)) # .to(device_loss)
            m_scaler.scale(m_loss / grad_step_acc).backward()
            
            m_msg_tqdm = conv2tqdm_msg(m_msg)
            progress_bar.set_postfix(m_msg_tqdm)
            # add to the avg metrics
            # and to the log
            t = []
            for key in m_msg:
                if key not in avg_metrics:
                    avg_metrics[key] = 0
                avg_metrics[key] += m_msg[key]
                t.append(m_msg[key])
            if m_sch:
                t.append(m_sch.get_last_lr()[0])
            m_loss_log.append(t)

                
            # Update the discriminator
            with torch.cuda.amp.autocast(dtype = fp16):
                batch_recon, q_loss, perplexity_s, _, _ = model(batch, encoder_tanh)
                d_loss, d_msg = LPIPSDiscLoss(q_loss.to(device_loss), batch.to(device_loss),
                                              batch_recon.to(device_loss), 1, step,
                                              last_layer = model.get_last_layer().to(device_loss))
            d_scaler.scale(d_loss / grad_step_acc).backward()
            t = []
            for key in d_msg:
                t.append(d_msg[key])
            else:
                t.append(config['optimizer']['lr'])
            if d_sch:
                t.append(d_sch.get_last_lr()[0])
            else:
                t.append(config['optimizer_d']['lr'])
            d_loss_log.append(t)


            # update scaler, optimizer, and backpropagate
            if step != 0 and step % grad_step_acc == 0  or (bstep+1) == len(train_loader):
                m_scaler.step(m_opt)
                m_scaler.update()
                if config['lr_scheduler']['type'].lower() == 'onecyclelr':
                    if m_sch:
                        try:
                            m_sch.step() 
                        except Exception as e:
                            print(e)
                m_opt.zero_grad(set_to_none = True)

                d_scaler.step(d_opt)
                d_scaler.update()
                if config['lr_scheduler_d']['type'].lower() == 'onecyclelr':
                    if d_sch:
                        try:
                            d_sch.step() 
                        except Exception as e:
                            print(e)
                d_opt.zero_grad(set_to_none = True)
                
            total_loss.append(m_loss.item())    
            avg_tot += total_loss[-1]           

            # save generated images
            if step != 0 and (step+1) % sample_every == 0:
                with torch.no_grad():
                    x_recon, _, _, _, _ = model(x_original.to(device_model))
                    #enc_x = model.encode(x_original.to(device_model))

                x_recon = unscale_tensor(x_recon)
                save_grid_imgs(x_recon, max(x_recon.shape[0] // 4, 2), \
                                f'{img_folder}/recon_imgs-{step+1}-{epoch+1}.jpg')
                
            # save checkpoints
            if step != 0 and (step+1) % save_every == 0:
                #checkpoint = {
                #    'model_orig': model_eager.state_dict(),
                #    'm_opt': m_opt.state_dict(),
                #    'm_scaler': m_scaler.state_dict(),
                #    'd_opt': d_opt.state_dict(),
                #    'd_scaler': d_scaler.state_dict()
                #}
                #torch.save(checkpoint, f'{chkpts}/chkpt_{step+1}-{epoch+1}.pt')
                torch.save(model_eager.state_dict(), f'{chkpts}/model_orig_{step+1}-{epoch+1}.pt')
                # save the discriminator
                save_model(discriminator, disc_name)
            
            # Update the global step
            step += 1
            
            # save the losses
            t = np.array(m_loss_log)
            np.savetxt(m_loss_fname, t, header =','.join(list(m_msg.keys()) + ['lr']), delimiter="," )
            t = np.array(d_loss_log)
            np.savetxt(d_loss_fname, t, header = ','.join(list(d_msg.keys()) + ['lr']), delimiter="," )
            
        # save a snapshot
        if save_snapshot:
            torch.save(model_eager.state_dict(), f'{chkpts}/model_orig_snapshot.pt')
        
        # save the discriminator
        save_model(discriminator, disc_name)

        # update scheduler (MultiStepLR)
        if config['lr_scheduler']['type'].lower() == 'multisteplr':
            if m_sch:
                m_sch.step() 
        if config['lr_scheduler_d']['type'].lower() == 'multisteplr':
            if d_sch:
                d_sch.step() 
            
        avg_tot /= (bstep+1)
        #msg = f'\t----> loss: {avg_tot:<2.5f}'
        print(f'Epoch {epoch+1} in {time.time() - t_start:.2f}')
        #print(msg)
        try:
            msg = '\t---->'
            for key in avg_metrics:
                if key != 'Step':
                    msg += f' {key}: {avg_metrics[key]/(bstep + 1):>.5f};'
            print(msg)
        except Exception as e:
            print(e)
        
    torch.save(model_eager.state_dict(), f'{chkpts}/final_model_{epoch+1}.pt')
    print(f'Saved the final modet at\n\t{chkpts}/final_model_{epoch+1}.pt')
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
