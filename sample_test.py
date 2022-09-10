import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"

from training import *

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

channels = [32,64, 128]
#ch_mults = [  [1,2,2],  [1,1,1,1], [1,1,1,2], [1,1,2,2], [1,2,2,2],[1,2,2,4],[1,2,4,4],[1,2,4,8]]
ch_mults = [[1,2,2,2]]
num_res = [0,1,2][::-1]
clamps = [3]
beta_mins =[0.11]
lrs = [ 1e-4]
x = [channels, ch_mults, num_res, clamps, beta_mins, lrs]



def tuning(x):
 channels = x[0]
 ch_mults = x[1]
 num_res = x[2]
 clamps = x[3]
 beta_mins =x[4]
 lrs = x[5]
 """
 for ch in channels:
    for ch_mult in ch_mults:
     for lr in lrs:
        for num_res_blocks in num_res:
         for training_clamp in clamps:
          for beta_min in beta_mins:
            train(alpha=1.9, lr=lr, batch_size=64, beta_min=beta_min, beta_max=7.5,
                  n_epochs=20, num_steps=1000, datasets='CIFAR10', path=None, device='cuda',
                  training_clamp=training_clamp, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, dropout=0.1, initial_epoch=0)
tuning(x)
"""
train(alpha=1.9, lr=5e-05, batch_size=64, beta_min=0.05, beta_max=7.5,
       n_epochs=20, num_steps=1000, datasets='CIFAR10', device='cuda',
       training_clamp=3.5, ch=32, ch_mult=[1,2,2], num_res_blocks=2, dropout=0.1,
       initial_epoch=19, path="/scratch/private/eunbiyoon/Levy_motion-/CIFAR10clamp3.5batch64lr5e-05ch32ch_mult[1, 2, 2]num_res2dropout0.11.9_0.05_7.5/ckpt/epoch19CIFAR10batch64ch32ch_mult[1, 2, 2]num_res2dropout0.11.9_0.05_7.5.pth")