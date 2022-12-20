import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6"

import random
import torch.backends.cudnn as cudnn
import torch
import numpy as np
a=0
torch.manual_seed(a)
torch.cuda.manual_seed(a)
torch.cuda.manual_seed_all(a)
np.random.seed(a)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(a)

from sampling import *
from training import *
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

import random
import torch.backends.cudnn as cudnn


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

#1.35
for epoch in torch.arange(19,20):
 path = f'/scratch/private/eunbiyoon/Levy_motion-/CIFAR10noattch4clamp3.01.9_0.1_7.5/ckpt/CIFAR10noattch4clamp3.0_epoch{epoch}_1.9_0.1_7.5.pth'
 samples = sample(alpha=1.5,path='/home/eunbiyoon/comb_Levy_motion/approximationCelebAbatch128lr0.0001ch128ch_mult[1, 2, 2, 2, 4]num_res2dropout0.1clamp501.5_0.1_15/ckpt/CelebAbatch128ch128ch_mult[1, 2, 2, 2, 4]num_res2dropout0.1clamp50_epoch0_1.5_0.1_15.pth',
                   beta_min=0.1, beta_max=15, sampler='ode_sampler', batch_size=64, num_steps=20, LM_steps=50,
                   Predictor=True, Corrector=False, trajectory=False, clamp=50, initial_clamp=5.5, clamp_mode="constant",
                   datasets="CelebA", name=str(epoch.item()), mode ='approximation',
                 dir_path='/home/eunbiyoon/comb_Levy_motion')


