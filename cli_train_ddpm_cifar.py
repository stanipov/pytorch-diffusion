#!/home/sf/data/linux/pyenv/pt1/bin/python
import time, os, json

import torch
from torch.utils.data import Subset
import numpy as np

from tqdm import tqdm

# to avoid IO errors with massive reading of the files
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from torch import optim
import torchvision
import torchvision.transforms as transforms

from src.utils.aux import *
from src.models.conv_blocks import *
from src.models.helpers import *
from src.models.noise_schedulers import *
from src.models.attention import *
from src.models.unet import Unet
from src.models.ddpm import Diffusion

# ===================================================================    
def get_num_params(m):
    return sum(p.numel() for p in m.parameters())

# ===================================================================    
def main(config_file):
    
    # TORCH_CUDNN_V8_API_ENABLED=1 
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True

    with open(config_file, 'r') as f:
        config = json.load(f)

    # model name
    model_name = config['model']['name']

    # results and chkpts folders:
    cwd = config['results']['root']
    if cwd == 'cwd':
        cwd = os.getcwd()
    results_folder = os.path.join(cwd, *('results', model_name))
    chkpts         = os.path.join(cwd, *('results', model_name, 'chkpts'))
    os.makedirs(chkpts, exist_ok=True)

    # Dataset params
    image_size = config['dataset']['image_size']
    root = './cifar10'
    use_subset = config['dataset']['use_subset']
    if use_subset:
        use_subset = float(use_subset)
        
    # Model params   
    load_name = config['model']['load_name']
    load      = config['model']['load']
    channels  = config['model']['channels']
    init_dim  = config['model']['init_dim']
    dim_mults = config['model']['dim_mults']
    timesteps = config['model']['timesteps']
    if config['model']['noise_scheduler'] == 'cosine':
        noise_scheduler = cosine_beta_schedule
    else:
        raise("Not implemented")


    # training params
    if  'bfloat' in config['training']['fp16']:
        fp16 = torch.bfloat16
    if config['training']['fp16'] == 'fp16':
        fp16 = torch.float16
    if not config['training']['fp16']:
        fp16 = config['training']['fp16']
    num_epochs         = int(config['training']['num_epochs'])
    batch_size         = int(config['training']['batch_size'])
    dataloader_workers = int(config['training']['dataloader_workers'])
    sample_every       = int(config['training']['sample_every'])
    save_every         = int(config['training']['save_every'])
    save_snapshot      = config['training']['save_snapshot']


    # optimizer
    lr    = float(config['optimizer']['lr'])
    betas = config['optimizer']['betas']
    eps   = float(config['optimizer']['eps'])


    # lr_scheduler
    lr_scheduler_enabled = config['lr_scheduler']['enabled']
    milestones           = config['lr_scheduler']['milestones']
    gamma                = float(config['lr_scheduler']['gamma'])


    print('Setting the dataset')
    if channels == 1:
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                        transforms.ToTensor(), 
                                        transforms.Lambda(lambda t: (t * 2) - 1)])
    else:
        transform = transforms.Compose([transforms.ToTensor(), 
                                        transforms.Lambda(lambda t: (t * 2) - 1)])       
                                  
    trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                        download=True, transform=transform)
    
    if use_subset:
        print(f'\t{use_subset} of the total dataset will be used')
        trainset = Subset(trainset, range(0, int(len(trainset)*use_subset)))
    else:
        print(f'Using whole dataset of {len(trainset)}')
                                   
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=dataloader_workers)
                                        

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'{device} will be used')
    print('Setting the model')
    
    # set the model
    model = Unet(
        dim = image_size,
        channels = channels,
        init_dim = init_dim,
        dim_mults = dim_mults
    )
    
    if load:
        PATH = f'{chkpts}/{load_name}'
        print(f'\tLoading the model from\n\t{PATH}')
        model.load_state_dict(torch.load(PATH))
        print('\tDone')
    
    print(f'Model parameters: {get_num_params(model):,}')
    model = model.to(device)

    ddpm = Diffusion(noise_scheduler, model, timesteps)
    optimizer = optim.AdamW(params = model.parameters(),lr = lr, betas=betas, eps = eps,  )
    

    scheduler = None
    if lr_scheduler_enabled:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                           milestones=milestones, 
                                           gamma=0.75, last_epoch=-1, verbose=True)
    
    

    
    if save_snapshot:
        print(f'Model snapshots will be saved in {chkpts}')
    

    # set up the optimizer, scheduler, and AMP scaler
    if fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    train_losses = []
    step_losses = []
    print(f'Training for {num_epochs} epochs')
    print(f'Sampling every {sample_every} epochs')
    print(f'Saving every {save_every} epochs')
    print('---------------------------\n\t\tTraining\n---------------------------\n')
    step = 1
    for epoch in range(num_epochs):
        train_loss = 0
        #for X in tqdm(train_loader, desc=f'Train {epoch}', total = len(train_loader), leave=True):
        progress_bar = tqdm(train_loader, desc=f'Train {epoch+1}', total = len(train_loader), leave=False, disable=False)
        for X in progress_bar:
            optimizer.zero_grad()

            batch_size = X[0].shape[0]
            batch = X[0].to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = ddpm.loss(batch, t, loss_type="huber", fp16=fp16) # 'huber'

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            step_losses.append(loss.item())
            #tqdm.write(f'\t----->Batch loss: {loss.item():.5f}')
            progress_bar.set_postfix({f'Step {step} loss' : f'\t{loss.item():.5f}'})
            step += 1

        if scheduler:
            scheduler.step()        
        
        # save generated images and checkpoints
        if epoch != 0 and (epoch+1) % sample_every == 0:
            samples = ddpm.sample(image_size=image_size, batch_size = 64, channels=channels)
            all_images = unscale_tensor(samples[-1])
            save_grid_imgs(all_images, all_images.shape[0] // 8, f'{results_folder}/sample-{epoch+1}.png')
            torch.save(model.state_dict(), f'{chkpts}/model_sampled-at_{epoch+1}.pt')

        if epoch != 0 and (epoch+1) % save_every == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict()
            }
            torch.save(checkpoint, f'{chkpts}/chkpt_{epoch+1}.pt')
            torch.save(model.state_dict(), f'{chkpts}/model_fp32_{epoch+1}.pt')
           
        if save_snapshot:
            torch.save(model.state_dict(), f'{chkpts}/model_snapshot.pt')

        tqdm.write(f'\t-->Epoch {epoch+1} mean loss: {train_loss/len(train_loader):.5f}')
        train_losses.append(train_loss/len(train_loader))
        
        np.savetxt(f'{chkpts}/avg_loss.dat', np.array(train_losses))
        np.savetxt(f'{chkpts}/step_loss.dat', np.array(step_losses))

    torch.save(model.state_dict(), f'{chkpts}/final_model_{epoch}.pt')
    #rain_losses = np.array(train_losses)
    #np.savetxt(f'{chkpts}/avg_loss.dat', train_losses)
    #step_losses = np.array(step_losses)
    #np.savetxt(f'{chkpts}/step_loss.dat', step_losses)

# ===================================================================    
if __name__ == '__main__':
    import argparse
    
    arg_desc = '''\
        Training of a simple diffusion model in CIFAR10 dataset
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
