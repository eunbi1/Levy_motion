from training import *
from sampling import *
import numpy as np
alpha=1.9

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

for beta_min in np.linspace(0.1,1, 10)[::-1]:
    for beta_max in np.linspace(10,20,10)[::-1]:
        ckpt_name = str(f'{alpha}_{beta_min}_{beta_max}.pth')
        train(alpha=alpha,beta_min=beta_min, beta_max=beta_max, n_epochs=100)
        sample(path=ckpt_name, alpha=alpha, beta_min=beta_min, beta_max = beta_max)