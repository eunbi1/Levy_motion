import torch
import copy
import time
import numpy as np
from scipy.special import gamma
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def get_continuous_time(index):
    return (index + 1) / 1000


def gamma_fn(x):
    return torch.tensor(gamma(x))


def get_continuous_time(index):
    return index / 1000


def get_discrete_time(t, N):
    return t * N


def gamma_fn(x):
    return torch.tensor(gamma(x))

from levy_stable_pytorch import LevyStable
levy = LevyStable()

def loss_fn(model, sde,
            x0: torch.Tensor,
            t: torch.LongTensor,
            e_L: torch,
            num_steps=1000):
    sigma = sde.marginal_std(t)
    x_coeff = sde.diffusion_coeff(t)

    if sde.alpha == 2:
        sigma_score = -1 / 2 * (e_L)

    else:
        sigma_score = levy.score(e_L, sde.alpha).to(device)

    x_t = x_coeff[:, None, None, None] * x0 + e_L * sigma[:, None, None, None]

    output = model(x_t, get_discrete_time(t, num_steps))
    weight = (sigma[:, None, None, None] * output - sigma_score)

    return (weight).square().sum(dim=(1, 2, 3)).mean(dim=0)  # +torch.abs(weight).sum(dim=(1,2,3)).mean(dim=0)



