import os
from sampling import *
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
import random
from torchvision.datasets import MNIST, CIFAR10, CelebA, CIFAR100
import tqdm
import os
import matplotlib.pyplot as plt
from models.ncsnpp import NCSNpp
import time
from model import *
from Diffusion import *
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

from transformers import  AdamW, get_scheduler
from torchvision import datasets, transforms
from torchlevy import LevyStable

levy = LevyStable()
def image_grid(x):
  size = 32
  channels = 3
  img = x.reshape(-1, size, size, channels)
  w = int(np.sqrt(img.shape[0]))
  img = img.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
  return img

def show_samples(x,name):
  x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
  img = image_grid(x)
  plt.figure(figsize=(8,8))
  plt.axis('off')
  plt.imshow(img)
  plt.savefig(name)
  plt.show()



def impainted_noise(sde, data, noise, mask,t):

    sigma = sde.marginal_std(t)

    x_coeff = sde.diffusion_coeff(t)

    data = x_coeff[:, None, None, None] * data +  noise* sigma[:, None, None, None]
    return data*mask+noise*(1-mask)


def impainted_ddim_score_update2(score_model, sde, data, mask, x_s, s, t, y=None, h=0.8, clamp=10, device='cuda', mode='approximation',
                       order=0):
    if y is not None:
        y = torch.ones((x_s.shape[0],)) * y

    alpha = sde.alpha
    score_s = score_model(x_s, s, y) * torch.pow(sde.marginal_std(s) + 1e-8, -(alpha - 1))[:, None, None, None]

    time_step = s - t
    beta_step = sde.beta(s) * time_step

    x_coeff = 1 + beta_step / alpha

    if alpha == 2:
        e_L = torch.randn(size=(x_s.shape)) * np.sqrt(2)
        e_L = e_L.to(device)
    else:
        e_L = levy.sample(alpha, 0, size=(x_s.shape), is_isotropic=True, clamp=20).to(device)

    #e_L = impainted_noise(sde, data,e_L,mask,s)

    if alpha == 2:
        score_coeff = beta_step * 2
    noise_coeff = torch.pow(beta_step, 1 / alpha)
    if order == 0:
        score_coeff = alpha * beta_step

    x_t = x_coeff[:, None, None, None] * x_s + score_coeff[:, None, None, None] * score_s + noise_coeff[:, None, None,
                                                                                            None] * e_L
    print('x_s range', torch.min(x_s), torch.max(x_s))
    print('x_t range', torch.min(x_t), torch.max(x_t))
    print('x_s mean', torch.mean(x_s))
    print('x_t mean', torch.mean(x_t))
    print('score range',torch.min(score_s), torch.max(score_s))

    return impainted_noise(sde, data, x_t, mask, t)


def impainted_pc_sampler2(score_models,
                sde,  data, mask,
                batch_size,
                num_steps,
                LM_steps=200,
                device='cuda',
                eps=1e-4,
                x_0=None,
                Predictor=True, mode='approximation',
                Corrector=False, trajectory=False,
                clamp=10,
                initial_clamp=10, final_clamp=1, y=None,
                datasets="CIFAR10", clamp_mode='constant', h=0.9):
    t = torch.ones(batch_size, device=device) * sde.T
    alpha = sde.alpha
    if datasets == "MNIST":
        if alpha < 2:
            x_s = levy.sample(alpha, 0, size=(batch_size, 1, 28, 28), is_isotropic=True, clamp=20).to(device)
        else:
            x_s = torch.randn(size=(batch_size, 1, 28, 28)) * np.sqrt(2)
            x_s = x_s.to(device)
    elif datasets == "CIFAR10":
        if alpha < 2:
            x_s = levy.sample(alpha, 0, size=(batch_size, 3, 32, 32), is_isotropic=True, clamp=20).to(device)
        else:
            x_s = torch.randn(size=(batch_size, 3, 32, 32)) * np.sqrt(2)
            x_s = x_s.to(device)
    elif datasets == "CelebA":
        if mode == "approximation" or "normal":
            x_s = torch.clamp(levy.sample(alpha, 0, size=(batch_size, 3, 64, 64)).to(device), -initial_clamp,
                              initial_clamp)
        if mode == "resampling":
            x_s = levy.sample(alpha, 0, clamp=initial_clamp, size=(batch_size, 3, 64, 64)).to(device)
        if mode == 'brownian':
            x_s = torch.clamp(levy.sample(1.5, 0, size=(batch_size, 3, 64, 64)).to(device), -initial_clamp,
                              initial_clamp)
    x_s = impainted_noise(sde, data,x_s,mask,t)
    if clamp_mode == "constant":
        linear_clamp = clamp
    if clamp_mode == "linear":
        linear_clamp = batch_time_step_t[0] * (clamp - final_clamp) + final_clamp
    if clamp_mode == "root":
        linear_clamp = torch.pow(batch_time_step_t[0], 1 / 2) * (clamp - final_clamp) + final_clamp
    if clamp_mode == "quad":
        linear_clamp = batch_time_step_t[0] ** 2 * (clamp - final_clamp) + final_clamp

    if trajectory:
        samples = []
        samples.append(x_s)
    time_steps = torch.pow(torch.linspace(sde.T, 1e-5, num_steps), 1)
    step_size = time_steps[0] - time_steps[1]

    batch_time_step_s = torch.ones(x_s.shape[0]) * time_steps[0]
    batch_time_step_s = batch_time_step_s.to(device)

    with torch.no_grad():
        for t in tqdm(time_steps[1:]):
            batch_time_step_t = torch.ones(x_s.shape[0]) * t
            batch_time_step_t = batch_time_step_t.to(device)

            if Predictor:
                x_s = impainted_ddim_score_update2(score_models, sde, data, mask, x_s, batch_time_step_s, batch_time_step_t, y=y,
                                         clamp=linear_clamp)
            if trajectory:
                samples.append(x_s)

            batch_time_step_s = batch_time_step_t

    if trajectory:
        return samples
    else:

        return x_s



image_size = 32
channels = 1
batch_size = 64
num_workers=0
alpha = 1.8
device = 'cuda'
path = "/scratch/private/eunbiyoon/sub_Levy/CIFAR10batch64lr0.0001ch128ch_mult[1, 2, 2, 2]num_res4dropout0.1clamp201.8_0.1_20/ckpt/CIFAR10batch64ch128ch_mult[1, 2, 2, 2]num_res4dropout0.1clamp20_epoch395_1.8_0.1_20.pth"
transform = Compose([Resize(image_size),ToTensor()])

dataset = CIFAR10('.', train=True, transform=transform, download=True)
data_loader = DataLoader(dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers, generator=torch.Generator(device=device))
batch = next(iter(data_loader))
data = batch[0].to(device)
show_samples(data, 'original_image.png')

mask = torch.ones_like(data).to(device)
mask[:, :, :, 16:] = 0.
show_samples(data * mask, 'masked_image.png')

sde = VPSDE(alpha=alpha)
ch=128
ch_mult=[1, 2, 2,2]
num_res_blocks=4
resolution=32
score_model = NCSNpp(ch=ch, ch_mult=ch_mult, resolution=resolution, num_res_blocks=num_res_blocks)

ckpt = torch.load(path, map_location=device)
score_model.load_state_dict(ckpt,strict=False)
score_model.to(device)
score_model.eval()

data = 2*data-1
x =  impainted_pc_sampler2(score_model,
                sde, data, mask,
                batch_size=batch_size,
                num_steps=1000)
x = (x+1)/2
x = x.clamp(0.0, 1.0)
show_samples(x,'impainted_image.png')
