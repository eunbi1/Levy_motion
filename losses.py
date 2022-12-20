import torch
import copy
import time
import numpy as np
import tqdm
from scipy.special import gamma
import torchlevy
from torchlevy import LevyStable
import torch.nn.functional as F

levy = LevyStable()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def gamma_func(x):
    return torch.tensor(gamma(x))

from torchlevy.approx_score import get_approx_score, rectified_tuning_score,generalized_gaussian_score




# def cosine_beta_schedule(timesteps, s=0.008):
#     """
#     cosine schedule as proposed in https://arxiv.org/abs/2102.09672
#     """
#     steps = timesteps + 1
#     x = torch.linspace(0, timesteps, steps)
#     alphas_cumprod = torch.cos((((x+1) / timesteps/0.9946) + s) / (1 + s) * torch.pi * 0.5) ** 2
#     alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
#     betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
#     return torch.clip(betas, 0.0001, 0.9999)
#
# # define beta schedule
# betas = cosine_beta_schedule(timesteps=1000)
#
# # define alphas
# alphas = 1. - betas
# alphas_cumprod = torch.cumprod(alphas, axis=0)
# alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
# sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
#
# # calculations for diffusion q(x_t | x_{t-1}) and others
# sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
# sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

def loss_fn(model, sde,
            x0: torch.Tensor,
            t: torch.LongTensor,
            y,
            e_L: torch,
            num_steps=1000, type="cft", training_clamp=4, mode='approximation'):
    sigma = sde.marginal_std(t)
    x_coeff = sde.diffusion_coeff(t)

    if sde.alpha == 2:

      score = -1 / 2 * (e_L)

    else:
        if mode =='approximation':
         #score =  levy.score(e_L, sde.alpha).to(device)



         score = -e_L/sde.alpha


    # t=t.long()
    # sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t]
    # sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t]
    x_t = x_coeff[:, None, None, None] * x0 + e_L * sigma[:, None, None, None]
    # x_t =sqrt_alphas_cumprod_t[:, None, None, None]  * x0 + sqrt_one_minus_alphas_cumprod_t[:, None, None, None]  * e_L
    # t = (t + 1) / 1000/0.9946
    output = model(x_t, t,y)
    # loss = torch.abs(weight).sum(dim=(1,2,3)).mean(dim=0)
    #
    # print('x_t', torch.min(x_t), torch.max(x_t))
    # print('e_L', torch.min(e_L),torch.max(e_L))
    # print('score', torch.min(score), torch.max(score))
    #print('output', torch.min(model(x_t, t)), torch.max(model(x_t, t)))
    #print('output*beta',torch.min(output), torch.max(output))
    #loss = F.smooth_l1_loss(output, score, size_average=False,reduce=True, beta=4.0)
    weight = output-score
    loss = (weight).square().sum(dim=(1, 2, 3)).mean(dim=0)
    #loss = F.smooth_l1_loss(output, score, size_average=False, reduce=True, beta=4.0)

    return  loss
