import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"

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

for epoch in torch.arange(19,20):
 path = f'/scratch/private/eunbiyoon/Levy_motion-/CIFAR10noattch4clamp3.01.9_0.1_7.5/ckpt/CIFAR10noattch4clamp3.0_epoch{epoch}_1.9_0.1_7.5.pth'
 samples = sample(alpha=1.9,path='/scratch/private/eunbiyoon/Levy_motion-/CIFAR10noatres1drop0.1clamp3.01.9_0.1_10.0/ckpt/CIFAR10noatres1drop0.1clamp3.0_epoch26_1.9_0.1_10.0.pth',
                   beta_min=0.1, beta_max=10, sampler='pc_sampler2', batch_size=64, num_steps=1000, LM_steps=50,
                   Predictor=True, Corrector=False, trajectory=False, clamp=3, initial_clamp=3, clamp_mode="constant",
                   datasets="CIFAR10", name=str(epoch.item()),
                 dir_path='/scratch/private/eunbiyoon/Levy_motion-/CIFAR10noatres1drop0.1clamp3.01.9_0.1_10.0')


