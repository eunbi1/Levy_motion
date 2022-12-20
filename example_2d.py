# -*- coding: utf-8 -*-

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn

from matplotlib.lines import Line2D
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm, trange



if not os.path.exists('figs'):
    os.mkdir('figs')

mu1 = np.array([5, 5])
sigma1 = np.array([[0.2, 0], [0, 0.2]])
X1 = np.random.multivariate_normal(mean=mu1, cov=sigma1, size=10000)

mu2 = np.array([-5, -5])
sigma2 = np.array([[0.2, 0], [0, 0.2]])
X2 = np.random.multivariate_normal(mean=mu2, cov=sigma2, size=1000)

X = np.concatenate([X1, X2])

plt.scatter(X[:-1, 0], X[:-1, 1], marker='.', color='k', alpha=0.1)
plt.scatter(X[-1, 0], X[-1, 1], marker='.', color='k', alpha=1.0, label=r'$\mathbf{x}^{(i)}$')#=\left(x^{(i)}, y^{(i)}\right)^{T}$')
plt.xlim(-11, 11)
plt.ylim(-11, 11)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.savefig(os.path.join('figs', '1_data.png'), dpi=300)
plt.show()

def score_loss(model, sde, 
               x0: torch.Tensor,
               t: torch.LongTensor,
               e: torch.Tensor):

    x_coeff = sde.diffusion_coeff(t)
    sigma = sde.marginal_std(t)

    x_t = x0 * x_coeff[:,None] + e * sigma.view(-1,1)

    if sde.alpha == 2:
        score = - 1 / 2 * torch.pow(sigma + 1e-4, -1).view(-1,1) * sde.beta(t).view(-1,1) * e

    output = model(x_t, t.float()) * sde.beta(t).view(-1,1)
    
    return (output - score).square().sum(dim=1).mean(dim=0)

import torch
import math

class VPSDE:
    def __init__(self, alpha, beta_min=0.1, beta_max=20, T=1.):
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.alpha = alpha
        self.T = T
    
    def beta(self, t):
        return (self.beta_1 - self.beta_0) * t + self.beta_0

    def marginal_log_mean_coeff(self, t):
        log_alpha_t = - 1 / (2 * self.alpha) * (t ** 2) * (self.beta_1 - self.beta_0) - 1 / self.alpha * t * self.beta_0
        return log_alpha_t

    def diffusion_coeff(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        return torch.pow(1. - torch.exp(self.alpha * self.marginal_log_mean_coeff(t)), 1 / self.alpha)

    def marginal_lambda(self, t):
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_sigma = torch.log(torch.pow(1. - torch.exp(self.alpha * log_mean_coeff), 1 / self.alpha))
        return log_mean_coeff - log_sigma

    def inverse_lambda(self, lamb):
        tmp = 2. * (self.beta_1 - self.beta_0) * torch.logaddexp(-2. * lamb, torch.zeros((1,)).to(lamb))
        Delta = self.beta_0**2 + tmp
        return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)

dataset = TensorDataset(torch.tensor(X).float())
dataloader = DataLoader(dataset, batch_size=256, pin_memory=True, shuffle=True)

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=2),nn.Linear(2,2))
        self.act = lambda x: x * torch.sigmoid(x)

        self.l1=torch.nn.Linear(2, 16)
        self.dense1= Dense(2,16)
        self.re=torch.nn.ReLU()
        self.l2=torch.nn.Linear(16, 16)
        self.dense2 = Dense(2,16)
        self.l3=torch.nn.Linear(16, 2)
        self.dense3 = Dense(2,2)


    def forward(self, x, t):

        embed = self.act(self.embed(t)) #2
        h1 = self.l1(x) 
        h1 += self.dense1(embed)
        h1 = self.act(h1)
  
        h2 = self.l2(h1)
        h2 += self.dense2(embed)
        h2 = self.act(h2)

        h3 = self.l3(h2)
        h3 += self.dense3(embed)
        h3 =self.act(h3)


        return h3

class MLP(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.model = torch.nn.Sequential(
                        torch.nn.Linear(2 + 1, 32),
                        torch.nn.ReLU(),
                        torch.nn.Linear(32, 32),
                        torch.nn.ReLU(),
                        torch.nn.Linear(32, 2)
                    )

                def forward(self, x, t):
                    t = 10*t - 5
                    x = torch.concat([x, t[:, None]], dim=-1)
                    return self.model(x)

import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
max_iter = 100
from scipy.stats import levy
step = 0
losses = []
X_noised = []

score_model = MLP()
optimizer = torch.optim.Adam(score_model.parameters(), amsgrad=True)
score_model.train()

sde = VPSDE(2, beta_min=0.1, beta_max=20, T=1.)

tqdm_epoch = tqdm.notebook.trange(max_iter)
L = []
for epoch in tqdm_epoch:
      avg_loss = 0.
      num_items = 0
      for i, x in enumerate(dataloader):
                x = x[0]
                n = x.size(0)
                step += 1
                e = torch.randn_like(x)
                t = torch.rand(n)

                loss = score_loss(score_model, sde, x,t,e) 
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]
      L.append(avg_loss/num_items)
      tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items)) 
      torch.save(score_model.state_dict(), 'ckpt.pth')
X = np.arange(epoch + 1)
plt.plot(X,L)

def figure( X):
        plt.scatter(X[:-1, 0], X[:-1, 1], marker='.', color='k', alpha=0.1)
        plt.scatter(X[-1, 0], X[-1, 1], marker='.', color='k', alpha=1.0,
                    label=r'$\mathbf{x}^{(i)}$')  # =\left(x^{(i)}, y^{(i)}\right)^{T}$')
        # plt.xlim(-11, 11)
        # plt.ylim(-11, 11)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.show()



def gamma_func(x):
    return torch.tensor(gamma(x))


def ddim_score_update2(score_model, sde, x_s, s, t, h=0.6, clamp = 10, device='cuda'):
    score_s = score_model(x_s, s) * torch.pow(sde.marginal_std(s) + 1e-4, -1)[:, None]
    time_step = s-t
    beta_step = sde.beta(s)*time_step
    x_coeff = 1 + beta_step/sde.alpha
    if sde.alpha==2:
        score_coeff2 = torch.pow(beta_step, 2 / sde.alpha) * 2

    else:
        score_coeff2 = torch.pow(beta_step, 2/sde.alpha)*torch.pow(time_step, 1-2/sde.alpha)*np.sin(torch.pi/2*(2-sde.alpha))/(2-sde.alpha)*2/torch.pi*gamma_func(sde.alpha+1)
    noise_coeff = torch.pow(beta_step, 1 / sde.alpha)

    e_L = torch.randn(size= x_s.shape)*np.sqrt(2)

    x_t = x_coeff[:, None] * x_s + score_coeff2[:, None] * score_s + noise_coeff[:, None] * e_L
    #print('score_coee', torch.min(score_coeff), torch.max(score_coeff))
    #print('noise_coeff',torch.min(noise_coeff), torch.max(noise_coeff))
    #print('x_coeff', torch.min(x_coeff), torch.max(x_coeff))
  

    #print('x_t range', torch.min(x_t), torch.max(x_t))
    #print('x coeff adding', torch.min(x_coeff[:, None, None, None] * x_s), torch.max(x_coeff[:, None, None, None] * x_s))
    #print('score adding',torch.min(score_coeff[:, None, None, None] * score_s), torch.max(score_coeff[:, None, None, None] * score_s) )
    #print('noise adding', torch.min(noise_coeff[:, None, None,None] * e_L), torch.max(noise_coeff[:, None, None,None] * e_L))

    return x_t


def pc_sampler2(score_model,
                sde,
                alpha,
                batch_size,
                num_steps,
                LM_steps=200,
                device='cuda',
                eps=1e-4,
                x_0= False,
                Predictor=True,
                Corrector=False, trajectory=False,
                clamp = 10,
                initial_clamp =3, final_clamp = 1,
                datasets="MNIST", clamp_mode = 'constant'):


    t = torch.ones(batch_size, device=device) 
    x_s = torch.randn(size= (batch_size, 2))*np.sqrt(2)



    time_steps = torch.linspace(1-1e-4, 0, num_steps)
    step_size = time_steps[0] - time_steps[1]

    batch_time_step_s = torch.ones(x_s.shape[0])
    batch_time_step_s = batch_time_step_s.to(device)

    with torch.no_grad():
        for t in tqdm.tqdm(time_steps):
            batch_time_step_t = torch.ones(x_s.shape[0])*t
            batch_time_step_t = batch_time_step_t

      
            x_s = ddim_score_update2(score_model, sde, x_s, batch_time_step_s, batch_time_step_t)
            batch_time_step_s = batch_time_step_t


    return x_s

x = pc_sampler2(score_model,
                sde,
                alpha=sde.alpha,
                batch_size=1000,
                num_steps=1000,
                LM_steps=200,
                device='cpu',
                eps=1e-4,
                x_0= False,
                Predictor=True,
                Corrector=False, trajectory=False,
                clamp = 10,
                initial_clamp =3, final_clamp = 1,
                datasets="MNIST", clamp_mode = 'constant')
figure(x)