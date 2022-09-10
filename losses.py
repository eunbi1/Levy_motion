import torch
import copy
import time
import numpy as np
import tqdm
from scipy.special import gamma
import torchlevy
from torchlevy import LevyStable
levy = LevyStable()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def gamma_func(x):
    return torch.tensor(gamma(x))






def loss_fn(model, sde,
            x0: torch.Tensor,
            t: torch.LongTensor,
            e_L: torch,
            num_steps=1000, type="backpropagation", mode='only'):
    sigma = sde.marginal_std(t)
    x_coeff = sde.diffusion_coeff(t)

    if sde.alpha == 2:
      score = -1 / 2 * (e_L)


    else:
        score = levy.score(e_L, sde.alpha, type=type).to(device)



    x_t = x_coeff[:, None, None, None] * x0 + e_L * sigma[:, None, None, None]
    output = model(x_t, t)
    weight = (output - score)
    #loss = torch.abs(weight).sum(dim=(1,2,3)).mean(dim=0)
    loss=(weight).square().sum(dim=(1, 2, 3)).mean(dim=0)
    #print('x_t', torch.min(x_t), torch.max(x_t))
    #print('e_L', torch.min(e_L),torch.max(e_L))
    #print('score', torch.min(score), torch.max(score))
    #print('output', torch.min(model(x_t, t)), torch.max(model(x_t, t)))
    #print('output*beta',torch.min(output), torch.max(output))
    #print('weight', torch.min(weight), torch.max(weight))

    return loss


def loss_fn(model, sde,
            x0: torch.Tensor,
            t: torch.LongTensor,
            e_L: torch,
            num_steps=1000, type="cft", mode='only'):
    sigma = sde.marginal_std(t)
    x_coeff = sde.diffusion_coeff(t)
    beta = sde.beta(t)[:,None,None,None]

    if sde.alpha == 2:
      #score = -1 / 2 * (e_L)*torch.pow(sigma+1e-5,-1)[:,None,None,None]*beta
      score = -1 / 2 * (e_L)


    else:
        #score = levy.score(e_L, sde.alpha, type=type).to(device)* torch.pow(sigma+1e-5,-1)[:,None,None,None]*beta
        score = levy.score(e_L, sde.alpha, type=type).to(device)



    x_t = x_coeff[:, None, None, None] * x0 + e_L * sigma[:, None, None, None]
    output = model(x_t, t)
    weight = (output - score)
    #loss = torch.abs(weight).sum(dim=(1,2,3)).mean(dim=0)
    loss=(weight).square().sum(dim=(1, 2, 3)).mean(dim=0)
    #print('x_t', torch.min(x_t), torch.max(x_t))
    #print('e_L', torch.min(e_L),torch.max(e_L))
    #print('score', torch.min(score), torch.max(score))
    #print('output', torch.min(model(x_t, t)), torch.max(model(x_t, t)))
    #print('output*beta',torch.min(output), torch.max(output))
    #print('weight', torch.min(weight), torch.max(weight))

    return loss
