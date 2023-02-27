#!/home/sf/data/linux/pyenv/pt1/bin/python
from src.models.vq_vae import VQ_VAE
from src.models.perceptual_loss import PerceptualLoss
from src.utils.aux import unscale_tensor, save_grid_imgs, get_num_params
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

# to avoid IO errors with massive reading of the files
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# ===================================================================    
def main(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
        
    # TORCH_CUDNN_V8_API_ENABLED=1 
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = True

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
    root = config['dataset']['location']
    use_subset = config['dataset']['use_subset']
    if use_subset:
        use_subset = float(use_subset)
    img_resize = config['dataset']['img_resize']


    # Model params   
    load_name       = config['model']['load_name']
    load            = config['model']['load']
    channels        = config['model']['channels']
    init_dim        = config['model']['init_dim']
    dim_mults       = config['model']['dim_mults']
    groups_grnorm   = config['model']['group_norm']
    resnet_stacks   = config['model']['resnet_stacks']
    embed_dim       = config['model']['embed_dim']
    commitment_cost = config['model']['commitment_cost']


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
        transform = transforms.Compose([(transforms.Grayscale(num_output_channels=1),transforms.Resize(image_size, interpolation='BICUBIC')) 
                                        if img_resize else transforms.Grayscale(num_output_channels=1),
                                        transforms.ToTensor(), 
                                        transforms.Lambda(lambda t: (t * 2) - 1)])
    else:
        transform = transforms.Compose([(transforms.Resize(image_size, interpolation='BICUBIC'), transforms.ToTensor()) 
                                        if img_resize else transforms.ToTensor(), 
                                        transforms.Lambda(lambda t: (t * 2) - 1)])       

    dataset = torchvision.datasets.ImageFolder(root = root, loader=Image.open, transform = transform)
    num_classes = len(dataset.classes)

    if use_subset:
        print(f'\t{use_subset} of the total dataset will be used')
        trainset = Subset(dataset, range(0, int(len(dataset)*use_subset)))
    else:
        print(f'Using whole of {len(dataset)} images')
    print(f'{num_classes} classes were found')

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=dataloader_workers)
    print('Done')

    # set the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'{device} will be used')
    print('Setting the model')
    model = VQ_VAE(
        img_ch          = channels, 
        enc_init_ch     = init_dim,
        ch_mult         = dim_mults, 
        grnorm_groups   = groups_grnorm,
        resnet_stacks   = resnet_stacks,
        embed_dim       = embed_dim, 
        commitment_cost = commitment_cost
    )

    if load:
        print(f'\tLoading the pretrined weights from\n\t{load_name}')
        model.load_state_dict(torch.load(load_name))
    print('Done')
    print(f'Model parameters: {get_num_params(model):,}')
    model = model.to(device)

    optimizer = optim.AdamW(params = model.parameters(),lr = lr, betas=betas, eps = eps,  )

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

    total_loss   = []
    quant_loss   = []
    perplexity   = []
    percept_loss = []

    ploss = PerceptualLoss(fp16).to(device)
    
    print(f'Training for {num_epochs} epochs')
    print(f'Sampling every {sample_every} epochs')
    print(f'Saving every {save_every} epochs')
    print('---------------------------\n\t\tTraining\n---------------------------\n')
    step = 1

    x_original = next(iter(train_loader))
    x_original = unscale_tensor(x_original[0])
    save_grid_imgs(x_original, x_original.shape[0] // 8, f'{results_folder}/original-images.jpg')

    for epoch in range(num_epochs):
        progress_bar = tqdm(train_loader, desc=f'Train {epoch+1}', total = len(train_loader), leave=False, disable=False)
        for X in progress_bar:
            optimizer.zero_grad()
            batch_size = X[0].shape[0]
            batch = X[0].to(device)

            with torch.cuda.amp.autocast(dtype = fp16):
                batch_recon, q_loss, perplexity_s, _, _ = model(batch)
                recon_loss = F.smooth_l1_loss(batch, batch_recon)

            p_loss = ploss(batch, batch_recon, 'huber')
            loss = q_loss + p_loss + recon_loss

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss.append(loss.item())
            quant_loss.append(q_loss.item())
            perplexity.append(perplexity_s.item())
            percept_loss.append(p_loss.item())    

            msg_dict = {
                f'Total {step} loss': f'{total_loss[-1]:.5f}',
                f'Quantization loss': f'{quant_loss[-1]:.5f}',
                f'Perceptual loss': f'{percept_loss[-1]:.5f}',
                f'Perplexity': f'{perplexity[-1]:.5f}',
            }    
            progress_bar.set_postfix(msg_dict)

            np.savetxt(f'{chkpts}/total_loss.dat', np.array(total_loss))
            np.savetxt(f'{chkpts}/quant_loss.dat', np.array(quant_loss))
            np.savetxt(f'{chkpts}/percept_loss.dat', np.array(percept_loss))
            np.savetxt(f'{chkpts}/perplex_loss.dat', np.array(perplexity))

        # save generated images and checkpoints
        if epoch != 0 and (epoch+1) % sample_every == 0:
            x_recon, _, _, _, _ = model(x_original)
            x_recon = unscale_tensor(x_recon)
            save_grid_imgs(x_recon, x_recon.shape[0] // 8, f'{results_folder}/recon_imgs-{epoch+1}.jpg')

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

        step += 1
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
