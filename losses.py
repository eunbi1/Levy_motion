import torch
import copy
import time
import numpy as np
from scipy.special import gamma
from score import *
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def get_continuous_time(index):
    return (index + 1) / 1000


# model은 [0,1]이지만 학습할 때는 Integer로 들어간다.
def get_discrete_time(t):
    return 1000. * torch.max(t - 1. / 1000, torch.zeros_like(t).to(t))


def get_beta_schedule(beta_start=0.0001, beta_end=0.02, num_timesteps=1000):
    return np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)


def gamma_fn(x):
    return torch.tensor(gamma(x))


def loss_fn(model, x0: torch.Tensor,
            t: torch.LongTensor,
            e_L: np.array,
            b: torch.Tensor,
            alpha, keepdim=False,
            approximation=True):
    aa = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    sigma = (torch.pow(1 - aa, 1 / alpha)).to(device)
    sigma = sigma.view(-1, 1, 1, 1)
    t0 = time.time()
    score = score_fn(alpha)

    sigma_score = score.evaluation(e_L)
    t1 = time.time()
    e_L = torch.Tensor(e_L).to(device)
    sigma_score = torch.Tensor(sigma_score).to(device)

    x_t = torch.pow(aa, 1 / alpha) * x0 + e_L * sigma
    output = model(x_t, t.float())
    weight = (sigma * output - sigma_score)

    return (weight).square().sum(dim=(1, 2, 3)).mean(dim=0)


