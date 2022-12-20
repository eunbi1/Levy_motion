import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="7"

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

import random
import torch.backends.cudnn as cudnn


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


# for epoch in torch.arange(19,20):
#  path = f'/scratch/private/eunbiyoon/Levy_motion-/CIFAR10noattch4clamp3.01.9_0.1_7.5/ckpt/CIFAR10noattch4clamp3.0_epoch{epoch}_1.9_0.1_7.5.pth'
#  samples = sample(alpha=1.8,path='/home/eunbiyoon/comb_Levy_motion/CelebAbatch128lr0.0001ch128ch_mult[1, 2, 2, 2, 4]num_res2dropout0.1clamp101.8_0.1_20/ckpt/CelebAbatch128ch128ch_mult[1, 2, 2, 2, 4]num_res2dropout0.1clamp10_epoch85_1.8_0.1_20.pth',
#                    beta_min=0.1, beta_max=20, sampler='ode_sampler', batch_size=9, num_steps=10, LM_steps=50, h=0.9,
#                    Predictor=True, Corrector=False, trajectory=False, clamp=1.5, initial_clamp=10, clamp_mode="constant",
#                    datasets="CelebA", name=str(epoch.item()), mode ='approximation',
#                  dir_path='/home/eunbiyoon/comb_Levy_motion')
#

# #
# for h in [0.2]:
#  for i in [3.5]:
#   name = str(h) +'_'+str(i)
#   samples = sample(alpha=1.5,path='/home/eunbiyoon/comb_Levy_motion/approximationCelebAbatch128lr0.0001ch128ch_mult[1, 2, 2, 2, 4]num_res2dropout0.1clamp501.5_0.1_15/ckpt/CelebAbatch128ch128ch_mult[1, 2, 2, 2, 4]num_res2dropout0.1clamp50_epoch180_1.5_0.1_15.pth',
#                    beta_min=0.1, beta_max=15, sampler='ode_sampler', batch_size=9, num_steps=20, LM_steps=5,
#                    Predictor=True, Corrector=False, trajectory=False, clamp=3.5, initial_clamp=50, clamp_mode="constant", h=h,
#                    datasets="CelebA", name=name, mode ='approximation',
#                  dir_path='/home/eunbiyoon/comb_Levy_motion')


#
for epoch in torch.arange(19,20):
 samples = sample(alpha=1.8,path='/scratch/private/eunbiyoon/sub_Levy/CIFAR10batch64lr0.0001ch128ch_mult[1, 2, 2, 2]num_res4dropout0.1clamp201.8_0.1_20/ckpt/CIFAR10batch64ch128ch_mult[1, 2, 2, 2]num_res4dropout0.1clamp20_epoch350_1.8_0.1_20.pth',
                   beta_min=0.1, beta_max=20, sampler='pc_sampler2', batch_size=16, num_steps=1000, LM_steps=20,
                   h=1.3,y=None,
                   Predictor=True, Corrector=False, trajectory=False, clamp=20, initial_clamp=20, clamp_mode="constant",
                   datasets="CIFAR10", name=str(epoch.item()), mode ='approximation',resolution=32, ch=128, ch_mult=[1, 2, 2,2], num_res_blocks=4,
                 dir_path='/scratch/private/eunbiyoon/sub_Levy')


# for epoch in torch.arange(19,20):
#  path = f'/scratch/private/eunbiyoon/Levy_motion-/CIFAR10noattch4clamp3.01.9_0.1_7.5/ckpt/CIFAR10noattch4clamp3.0_epoch{epoch}_1.9_0.1_7.5.pth'
#  samples = sample(alpha=1.5,path='/home/eunbiyoon/comb_Levy_motion/approximationCIFAR10batch128lr0.0001ch128ch_mult[1, 2, 2, 2]num_res2dropout0.1clamp1001.5_0.1_20/ckpt/CIFAR10batch128ch128ch_mult[1, 2, 2, 2]num_res2dropout0.1clamp100_epoch165_1.5_0.1_20.pth',
#                    beta_min=0.1, beta_max=20, sampler='ode_sampler', batch_size=64, num_steps=20, LM_steps=50,
#                    h=0.9,
#                    Predictor=True, Corrector=False, trajectory=False, clamp=3., initial_clamp=100, clamp_mode="constant",
#                    datasets="CIFAR10", name=str(epoch.item()), mode ='approximation',resolution=32, ch=128, ch_mult=[1, 2, 2,2], num_res_blocks=2,
#                  dir_path='/home/eunbiyoon/comb_Levy_motion')

# for epoch in torch.arange(19,20):
#  path = f'/scratch/private/eunbiyoon/Levy_motion-/CIFAR10noattch4clamp3.01.9_0.1_7.5/ckpt/CIFAR10noattch4clamp3.0_epoch{epoch}_1.9_0.1_7.5.pth'
#  samples = sample(alpha=1.5,path='/home/eunbiyoon/comb_Levy_motion/approximationCIFAR10batch128lr0.0001ch128ch_mult[1, 2, 2, 2]num_res2dropout0.1clamp2001.5_0.1_20/ckpt/CIFAR10batch128ch128ch_mult[1, 2, 2, 2]num_res2dropout0.1clamp200_epoch5_1.5_0.1_20.pth',
#                    beta_min=0.1, beta_max=20, sampler='ode_sampler', batch_size=30, num_steps=20, LM_steps=50,
#                    h=0.9,
#                    Predictor=True, Corrector=False, trajectory=False, clamp=3.5, initial_clamp=200, clamp_mode="constant",
#                    datasets="CIFAR10", name=str(epoch.item()), mode ='approximation',resolution=32, ch=128, ch_mult=[1, 2, 2,2], num_res_blocks=2,
#                  dir_path='/home/eunbiyoon/comb_Levy_motion')
