import os
from sampling import *
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
import tqdm
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
from levy_stable_pytorch import LevyStable
levy = LevyStable()

def train(alpha=2, lr=1e-4, beta_min=0.1, beta_max=20, batch_size=64,
          n_epochs=10, num_steps=1000, dataset='MNIST',num_workers = 0):
    sde = VPSDE(alpha, beta_min=beta_min, beta_max=beta_max)
    sde = sde
    score_model = torch.nn.DataParallel(ScoreNet(sde))
    score_model = score_model.to(device)

    if device == 'cuda':
        num_workers = num_workers
    else:
        num_workers = 0

    dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
    data_loader = DataLoader(dataset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers, generator=torch.Generator(device=device),)

    optimizer = Adam(score_model.parameters(), lr=lr)
    tqdm_epoch = tqdm.trange(n_epochs)

    L = []
    counter = 0
    t_0 = time.time()
    for epoch in tqdm_epoch:
        counter += 1
        avg_loss = 0.
        num_items = 0
        for x, y in data_loader:
            x = x.to(device)
            e_L = levy.sample(alpha, 0, size=x.shape ).to(device)
            #e_L = levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=1, size=x.shape)
            #e_L = torch.Tensor(e_L).to(device)
            t = torch.rand(x.shape[0]).to(device)
            loss = loss_fn(score_model, sde, x, t, e_L, num_steps=num_steps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

        L.append(avg_loss / num_items)

        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        ckpt_name = str(f'{alpha}_{beta_min}_{beta_max}.pth')
        torch.save(score_model.state_dict(),ckpt_name)
    name = str(time.strftime('%m%d_%H%M_', time.localtime(time.time()))) +'_'+ 'alpha'+str(f'{alpha}')+ 'beta'+str(f'{beta_min}')+ '_'+ str(
            f'{beta_max}') + '.pth'
    dir_path = os.path.join(os.getcwd(), 'chekpoint')
    if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
    name = os.path.join(dir_path, name)
    torch.save(score_model.state_dict(), name)
    t_1 = time.time()
    print('Total running time is', t_1 - t_0)
    plt.plot(np.arange(n_epochs), L)

#for beta_min in np.linspace(0.1, 2, 5):
 # for beta_max in np.linspace(10,20,5):
  #  print('beta_min, beta_max', beta_min, beta_max)
   # train(beta_min=beta_min, beta_max=beta_max, n_epochs=10)
    #sample(beta_min=beta_min, beta_max = beta_max)


#print(1.9,1, 20)

