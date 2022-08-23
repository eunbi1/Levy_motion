import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6,7"

from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
import tqdm
import os
from scipy.stats import levy_stable
import matplotlib.pyplot as plt

from score import *
from model import *
from losses import *
import numpy as np
import torch

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
# @title Training score model

score_model = torch.nn.DataParallel(Model())
score_model = score_model.to(device)


# parameter
alpha = 2
num_timesteps = 1000

beta_start = 0.0001
beta_end = 0.02,
betas = get_beta_schedule(beta_start=beta_start, beta_end=beta_end, num_timesteps=num_timesteps)
b = torch.from_numpy(betas).float().to(device)

n_epochs = 50  # @param {'type':'integer'}
batch_size = 64  # @param {'type':'integer'}
lr = 1e-4  # @param {'type':'number'}

# dataset setting
dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

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
        n = x.size(0)
        x = x.to(device)
        e_L = levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=1, size=x.shape)
        t = torch.randint(low=0, high=num_timesteps, size=(n // 2 + 1,)).to(device)
        t = torch.cat([t, num_timesteps - t - 1], dim=0)[:n]
        loss = loss_fn(score_model, x, t, e_L, b, alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]

    L.append(avg_loss / num_items)

    if counter % 1 == 0:
        print(counter, 'th', avg_loss / num_items)

    # Print the averaged training loss so far.
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
    # Update the checkpoint after each epoch of training.
    torch.save(score_model.state_dict(), 'ckpt.pth')
    torch.save(score_model.state_dict(), os.getcwd()+'/score_l2only.pth')
t_1 = time.time()
print('Total running time is', t_1 - t_0)
plt.plot(np.arange(n_epochs), L)