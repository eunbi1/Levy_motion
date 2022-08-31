import os
from sampling import *
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST
from tqdm import tqdm
import os
from scipy.stats import levy_stable
import matplotlib.pyplot as plt
import time
from model import *
from losses import *
from Diffusion import *
import numpy as np
import torch

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)
torch.multiprocessing.set_start_method('spawn')
from torchlevy import LevyStable

levy = LevyStable()

image_size = 28
channels = 1
batch_size = 128


def train(alpha=2, lr = 1e-4, batch_size=64, beta_min=0.1, beta_max = 20,
          n_epochs=15,num_steps=1000, datasets ='MNIST',path = None):
    sde = VPSDE(alpha, beta_min = beta_min, beta_max = beta_max)

    score_model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,))

    score_model = score_model.to(device)
    # score_model = torch.nn.DataParallel(score_model)

    if path:
      ckpt = torch.load(path, map_location=device)
      score_model.load_state_dict(ckpt,  strict=False)

    if datasets =="MNIST":
        dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
        data_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True)
    if datasets == "FashionMNIST":
        dataset = FashionMNIST('.', train=True, transform=transforms.ToTensor(), download=True)
        data_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True)

    optimizer = Adam(score_model.parameters(), lr=lr)


    L = []
    counter = 0
    t0 = time.time()
    for epoch in tqdm(range(n_epochs)):
        counter += 1
        avg_loss = 0.
        num_items = 0
        for i, batch in tqdm(enumerate(data_loader)):
            x = batch[0]
            x = 2*x - 1
            x = x.to(device)
            e_L = torch.clamp(levy.sample(alpha, 0, size=x.shape ).to(device),-10,10)

            t = torch.rand(x.shape[0]).to(device)+1e-4
            loss = loss_fn(score_model, sde, x, t, e_L, num_steps=num_steps)

            optimizer.zero_grad()
            loss.backward()
            #print(f'{epoch} th epoch {i} th step loss: {loss}')
            if torch.isnan(loss):
                loss = loss_fn(score_model, sde, x, t, e_L, num_steps=num_steps, type="cft")
                #print(f' cft {epoch} th epoch {i} th step loss: {loss}')
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        t1 = time.time()
        L.append(avg_loss / num_items)
        print(f'{epoch} th epoch loss: {avg_loss / num_items}')
        print('Running time:', t1-t0)
        ckpt_name = str(datasets)+ str(f'{alpha}_{beta_min}_{beta_max}.pth')
        torch.save(score_model.state_dict(),ckpt_name)
        #name = str(datasets)+'ckpt.pth'
        #torch.save(score_model.state_dict(), name)
    name = str(datasets)+str(time.strftime('%m%d_%H%M_', time.localtime(time.time()))) +'_'+ 'alpha'+str(f'{alpha}')+ 'beta'+str(f'{beta_min}')+ '_'+ str(
            f'{beta_max}') + '.pth'
    dir_path = os.path.join(os.getcwd(), 'chekpoint')
    if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
    name = os.path.join(dir_path, name)
    torch.save(score_model.state_dict(), name)
    plt.plot(np.arange(n_epochs), L)

