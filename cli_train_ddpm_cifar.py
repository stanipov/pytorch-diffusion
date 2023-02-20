#!/home/sf/data/linux/pyenv/pt1/bin/python
import time, os

import torch
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
def main():
    # model name
    model_name = 'ddpm-1-cifar10'

    # results and chkpts folders:
    cwd = os.getcwd()
    results_folder = os.path.join(cwd, *('results', model_name))
    chkpts = os.path.join(cwd, *('results', model_name, 'chkpts'))
    os.makedirs(chkpts, exist_ok=True)

    # Dataset params
    image_size = 32
    root = './cifar10'

    # Model params
    channels  = 3
    dim_mults = (1, 2, 4)
    timesteps = 100
    noise_scheduler = cosine_beta_schedule

    # training params
    fp16                  = True
    num_epochs            = 5000
    batch_size            = 128
    dataloader_workers    = 8
    sample_every = 50
    save_every = 1000

    # optimizer
    lr    = 5e-4
    betas = (0.9, 0.999) 
    eps   = 1e-8 

    print('Setting the dataset')
    transform = transforms.Compose([transforms.ToTensor(), 
                                    transforms.Lambda(lambda t: (t * 2) - 1)])
    trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                        download=True, transform=transform)
                                        
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=dataloader_workers)
                                        

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'{device} will be used')
    print('Setting the model')
    
    # set the model
    model = Unet(
        dim = image_size,
        channels = channels,
        dim_mults = dim_mults
    )
    print(f'Model parameters: {get_num_params(model):,}')
    model = model.to(device)

    ddpm = Diffusion(noise_scheduler, model, timesteps)
    optimizer = optim.AdamW(params = model.parameters(),lr = lr, betas=betas, eps = eps,  )

    # set up the optimizer, scheduler, and AMP scaler
    if fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    train_losses = []
    step_losses = []
    print(f'Training for {num_epochs} epochs')
    for epoch in range(num_epochs):
        train_loss = 0
        for X in tqdm(train_loader, desc=f'Train {epoch}', total = len(train_loader)):
            optimizer.zero_grad()

            batch_size = X[0].shape[0]
            batch = X[0].to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = ddpm.loss(batch, t, loss_type="huber", fp16=fp16)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            step_losses.append(loss.item())

        # save generated images and checkpoints
        if epoch != 0 and epoch % sample_every == 0:
            samples = ddpm.sample(image_size=image_size, batch_size = 64, channels=channels)
            all_images = unscale_tensor(samples[-1])
            save_grid_imgs(all_images, all_images.shape[0] // 8, f'{results_folder}/sample-{epoch}.png')

        if epoch != 0 and epoch % save_every == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict()
            }
            torch.save(checkpoint, f'{chkpts}/chkpt_{epoch}.pt')
            torch.save(model.state_dict(), f'{chkpts}/model_fp32_{epoch}.pt')

        tqdm.write(f'\t----->Mean loss: {train_loss/len(train_loader):.5f}')
        train_losses.append(train_loss)

    torch.save(model.state_dict(), f'{chkpts}/final_model_fp32_{epoch}.pt')
    train_losses = np.array(train_losses)
    np.savetxt(f'{chkpts}/avg_loss.dat', train_losses)
    step_losses = np.array(step_losses)
    np.savetxt(f'{chkpts}/step_loss.dat', step_losses)

# ===================================================================    
if __name__ == '__main__':
    main()
