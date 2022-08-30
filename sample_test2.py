from sampling import *
from training import *
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity

import os


#train(beta_min=0.1, alpha=1.9,beta_max=20,n_epochs=1000)
#train(beta_min=1, alpha=1.9, abeta_max=20,n_epochs=1000)
samples = sample(alpha=1.9, path ='/home/eunbiyoon/Levy_motion-/MNIST1.9_0.1_10.0.pth',
            beta_min=0.1, beta_max=10, sampler='pc_sampler2', batch_size=64, num_steps=1000,LM_steps=20,
            Predictor=True, Corrector=False, trajectory = True)
#sample(alpha=1.9, path ="/home/eunbiyoon/Levy_motion-/MNIST1.9_0.1_20.pth",beta_min=0.1, beta_max=20, sampler='pc_sampler2', batch_size=64,num_steps=1000,LM_steps=20,Predictor=True, Corrector=False)

import matplotlib.animation as animation

random_index = 53

fig =   plt.figure(figsize=(6, 6))
image_size = 28
channels = 1
batch_size = 64
ims = []
for i in range(0,len(samples),10):
    sample = samples[i]
    sample_grid = make_grid(sample, nrow=int(np.sqrt(batch_size)))
    plt.axis('off')
    im=plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    ims.append([im])

animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
animate.save('diffusion_1.9.gif')
plt.show()