import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import random
import torch.backends.cudnn as cudnn
import torch
import numpy as np


from sampling import *
from training import *
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

import random
import torch.backends.cudnn as cudnn

torch.manual_seed(4)
torch.cuda.manual_seed(4)
torch.cuda.manual_seed_all(4)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

for epoch in torch.arange(49,50):
  samples =sample(alpha=1.9, path='/scratch/private/eunbiyoon/Levy_motion-/CelebAatt+ema+batch64lr1e-05ch128ch_mult[1, 2, 2, 2]num_res2dropout0.1clamp3.51.9_0.1_7.5/ckpt/CelebAbatch64ch128ch_mult[1, 2, 2, 2]num_res2dropout0.1clamp3.5_epoch8_1.9_0.1_7.5.pth',
                   beta_min=0.1, beta_max=7.5, sampler='dpm_sampler', batch_size=64, num_steps=25, LM_steps=5,
               Predictor=True, Corrector=False, trajectory=False, clamp=3, initial_clamp=3,
               clamp_mode="constant", x_0=False, ch=128, ch_mult=[1,2,2,2], num_res_blocks=2,
               datasets="CelebA", name=str(str(f'2epoch{epoch}')), dir_path='/scratch/private/eunbiyoon/Levy_motion-')
  samples = sample(alpha=1.9,
                   path='/scratch/private/eunbiyoon/Levy_motion-/CelebAatt+ema+batch64lr1e-05ch128ch_mult[1, 2, 2, 2]num_res2dropout0.1clamp3.51.9_0.1_7.5/ckpt/CelebAbatch64ch128ch_mult[1, 2, 2, 2]num_res2dropout0.1clamp3.5_epoch8_1.9_0.1_7.5.pth',
                   beta_min=0.1, beta_max=7.5, sampler='pc_sampler2', batch_size=64, num_steps=2000, LM_steps=5,
                   Predictor=True, Corrector=False, trajectory=False, clamp=3, initial_clamp=3,
                   clamp_mode="constant", x_0=False, ch=128, ch_mult=[1, 2, 2, 2], num_res_blocks=2,
                   datasets="CelebA", name=str(str(f'2epoch{epoch}')),
                   dir_path='/scratch/private/eunbiyoon/Levy_motion-')



