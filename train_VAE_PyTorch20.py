#!/home/sf/data/linux/pyenv/pt2/bin/python
from src.datasets.artbench import artbench_hires, artbench256
from src.models.vq_vae import VQModel, set_VQModel
#from src.models.perceptual_loss import PerceptualLoss
from src.losses.perceptual_loss import PerceptualLoss
from src.losses.focal_frequency_loss import FocalFrequencyLoss
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

    # Dataset params
    image_size = config['dataset']['image_size']
    root = config['dataset']['location']
    use_subset = config['dataset']['use_subset']
    if use_subset:
        use_subset = float(use_subset)
    img_resize = config['dataset']['img_resize']


    # Model params   
    load_name       = config['model']['load_name']
    load            = config['model']['load']


    # training params
    if  'bfloat' in config['training']['fp16']:
        fp16 = torch.bfloat16
    if config['training']['fp16'] == 'fp16':
        fp16 = torch.float16
    if not config['training']['fp16']:
        fp16 = None
    perceptual_loss    = config['training']['perceptual_loss']
    use_focal_loss         =  config['training']['focal_loss']
    num_epochs         = int(config['training']['num_epochs'])
    grad_step_acc      = int(config['training']['grad_step_acc'])
    batch_size         = int(config['training']['batch_size'])
    dataloader_workers = int(config['training']['dataloader_workers'])
    sample_every       = int(config['training']['sample_every'])
    save_every         = int(config['training']['save_every'])
    save_snapshot      = config['training']['save_snapshot']
    percept_loss_factor = np.array(config['training']['percept_loss_factor']) if config['training']['percept_loss_factor'] else None
    quatization_loss_factor = np.array(config['training']['quatization_loss_factor']) if config['training']['quatization_loss_factor'] else None
    recon_loss_factor = np.array(config['training']['recon_loss_factor']) if config['training']['recon_loss_factor'] else None


    # optimizer
    lr    = float(config['optimizer']['lr'])
    betas = config['optimizer']['betas']
    eps   = float(config['optimizer']['eps'])


    # lr_scheduler
    lr_scheduler_enabled = config['lr_scheduler']['enabled']
    milestones           = config['lr_scheduler']['milestones']
    gamma                = float(config['lr_scheduler']['gamma'])


    print('Setting the dataset')
    if img_resize and image_size > 256:
        print(f'Using the original dataset with rescaling to {image_size} pix')
        dataset = artbench_hires(root, image_size=image_size)
    else:
        print('Using CIFAR10-like dataset of 256 pix')
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
    print(f'\t{num_classes} classes were found')

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=dataloader_workers,
                                              pin_memory = True)
    print('Done')

    # set the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'{device} will be used')
    print('Setting the model')
    model_eager = set_VQModel(config, load_name)
    print('Done')
    print(f'Model parameters: {get_num_params(model_eager):,}')
    print(f'Compiling model')
    model_eager = model_eager.to(device)
#    model = torch.compile(model_eager, mode="max-autotune")
    model = torch.compile(model_eager)        
#    model = model.to(device)
    
    
    # perceptual loss
    if perceptual_loss:
        ploss = PerceptualLoss(fp16).to(device)
        
    # use focal freq loss
    if use_focal_loss:
        focal_loss_fn = FocalFrequencyLoss(patch_factor = 16)
    
    optimizer = optim.AdamW(params = model.parameters(),lr = lr, betas=betas, eps = eps)

    scheduler = None
    if lr_scheduler_enabled:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                           milestones=milestones, 
                                           gamma=gamma, last_epoch=-1, verbose=True)

    if save_snapshot:
        print(f'Model snapshots will be saved in {chkpts}')

    # set up the optimizer, scheduler, and AMP scaler
    if fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None


    # schedule gradual change of loss wight factors
    print('Preparing loss factors schedules')
    T = np.linspace(0, num_epochs, num_epochs)
    q_loss_sch = cos_schedule(T, min(quatization_loss_factor), max(quatization_loss_factor), num_epochs) if quatization_loss_factor is not None else np.ones(num_epochs)
    np.savetxt(f'{chkpts}/q_loss_sch.dat', q_loss_sch)
    
    percep_loss_sch = cos_schedule(T, min(percept_loss_factor), max(percept_loss_factor), num_epochs) if percept_loss_factor is not None else np.ones(num_epochs)
    np.savetxt(f'{chkpts}/percep_loss_sch.dat', percep_loss_sch)
    
    rec_loss_sch = cos_schedule(T, min(recon_loss_factor), max(recon_loss_factor), num_epochs) if recon_loss_factor is not None else np.ones(num_epochs) 
    np.savetxt(f'{chkpts}/rec_loss_sch.dat', rec_loss_sch)
    
    print(f'Training for {num_epochs} epochs')
    print(f'Sampling every {sample_every} steps')
    print(f'Saving every {save_every} steps')
    print('==============================================================\n\t\tTraining\n==============================================================\n')
    
    total_loss   = []
    quant_loss   = []
    #perplexity   = []
    percept_loss = []
    focal = []
    step = 0
        
    x_original = next(iter(train_loader))
    x_original = x_original[0]
    save_grid_imgs(unscale_tensor(x_original), max(x_original.shape[0] // 4, 2), f'{img_folder}/original-images.jpg')
    
    for epoch in range(num_epochs):
        t_start = time.time()
        progress_bar = tqdm(train_loader, desc=f'Train {epoch+1}', total = len(train_loader),
                            mininterval = 1.0, leave=False, disable=False, colour = '#009966')

        avg_tot = 0
        avg_quant = 0
        avg_percep = 0
        #avg_perplex = 0
        avg_focal = 0

        for bstep, X in enumerate(progress_bar):
            optimizer.zero_grad()
            batch_size = X[0].shape[0]
            batch = X[0].to(device, non_blocking=False)

            with torch.cuda.amp.autocast(dtype = fp16):
                batch_recon, q_loss, perplexity_s, _, _ = model(batch)
                recon_loss = F.smooth_l1_loss(batch, batch_recon)
                if perceptual_loss:
                    p_loss = ploss(batch, batch_recon, 'huber')
                else:
                    p_loss = 0
                if use_focal_loss:
                    with torch.cuda.amp.autocast(dtype = fp16):
                        _tmp = focal_loss_fn(batch_recon.to(torch.float32), batch.to(torch.float32))
                        focal_loss = torch.nan_to_num(_tmp, nan = float('inf')).to(fp16) #posinf = float('inf'), neginf = -float('inf)
                else:
                    focal_loss = 0
                loss = (q_loss_sch[epoch]*q_loss + p_loss*percep_loss_sch[epoch] + recon_loss*rec_loss_sch[epoch] + focal_loss) / grad_step_acc
            
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
            quant_loss.append(q_loss.item() / grad_step_acc)
            #perplexity.append(perplexity_s.item())
            if perceptual_loss:
                percept_loss.append(p_loss.item() / grad_step_acc)  
            else:
                percept_loss.append(0)  
            if use_focal_loss:
                focal.append(focal_loss.item() / grad_step_acc)
            else:
                focal.append(0)

            avg_tot += total_loss[-1]
            avg_quant += quant_loss[-1]
            avg_percep += percept_loss[-1]
            #avg_perplex += perplexity[-1]
            avg_focal += focal[-1] 

            msg_dict = {
                f'Step {step} total': f'{total_loss[-1]:.5f}',
                f'Quant': f'{quant_loss[-1]*q_loss_sch[epoch]:.5f}',
                f'Focal': f'{focal[-1]:.5f}' if use_focal_loss else None,
                f'Percep': f'{percept_loss[-1]*percep_loss_sch[epoch]:.5f}',
                #f'Perplex': f'{perplexity[-1]:.2f}'
            }    
            progress_bar.set_postfix(msg_dict)
            
            np.savetxt(f'{chkpts}/total_loss.dat', np.array(total_loss))
            np.savetxt(f'{chkpts}/quant_loss.dat', np.array(quant_loss))
            if perceptual_loss:
                np.savetxt(f'{chkpts}/percept_loss.dat', np.array(percept_loss))
            #np.savetxt(f'{chkpts}/perplex_loss.dat', np.array(perplexity))
            if use_focal_loss:
                np.savetxt(f'{chkpts}/focal_loss.dat', np.array(focal))

            # save generated images and checkpoints
            if step != 0 and (step+1) % sample_every == 0:
                with torch.no_grad():
                    x_recon, _, _, _, _ = model(x_original.to(device))
                x_recon = unscale_tensor(x_recon)
                save_grid_imgs(x_recon, max(x_recon.shape[0] // 4, 2), f'{img_folder}/recon_imgs-{step+1}-{epoch+1}.jpg')

            if step != 0 and (step+1) % save_every == 0:
                checkpoint = {
                    'model_orig': model_eager.state_dict(),
                    'model_compiled': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict()
                }
                torch.save(checkpoint, f'{chkpts}/chkpt_{step+1}-{epoch+1}.pt')
                torch.save(model.state_dict(), f'{chkpts}/model_compiled_{step+1}-{epoch+1}.pt')
                torch.save(model_eager.state_dict(), f'{chkpts}/model_orig_{step+1}-{epoch+1}.pt')
            step += 1
            

        if save_snapshot:
            torch.save(model.state_dict(), f'{chkpts}/model_compiled_snapshot.pt')
            torch.save(model_eager.state_dict(), f'{chkpts}/model_orig__snapshot.pt')
        
        if scheduler:
            scheduler.step()   

        # clean up cache
        try:
            del batch_recon
        except Exception as e:
            print(e)
        try:            
            del perplexity_s
        except Exception as e:
            print(e)
        try:
            del x_recon
        except Exception as e:
            print(e)
        try:
            del batch
        except Exception as e:
            print(e)
        try:
            del focal_loss
        except Exception as e:
            print(e)    
        torch.cuda.empty_cache()
        
        avg_tot /= (bstep+1)
        avg_quant /= (bstep+1)
        avg_percep /= (bstep+1)
        #avg_perplex /= (bstep+1)
        avg_focal /= (bstep+1)
        #msg = f'\t----> loss: {avg_tot:<2.5f}, focal: {avg_focal:<2.5f}, quant: {avg_quant:<2.5f}, percept {avg_percep:<2.5f}, perplex: {avg_perplex:<3.2f}'
        msg = f'\t----> loss: {avg_tot:<2.5f}, focal: {avg_focal:<2.5f}, quant: {avg_quant:<2.5f}, percept {avg_percep:<2.5f}'
        print(f'Epoch {epoch+1} in {time.time() - t_start:.2f}')
        print(msg)
        time.sleep(0.2)       
        

    torch.save(model_eager.state_dict(), f'{chkpts}/final_model_{epoch+1}.pt')
    print(f'Saved the final modet at\n\t{chkpts}/final_model_{epoch+1}.pt')
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
    
    msg = """
    ==============================================================
       Training of VQ-VAE image compressor for latent diffusion
     on ArtBench dataset (https://github.com/liaopeiyuan/artbench)
    ==============================================================   
    """
    print(msg)
    main(cfg_file)
