import os

from model import *
from cifar10_model import *
from Diffusion import *
from sampler import *
#from models.ncsnpp import NCSNpp
import torch
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from models.ncsnpp import NCSNpp
image_size = 28
channels = 1
batch_size = 128

def image_grid(x):
  size = x.shape[1]
  channels = x.shape[-1]
  img = x.reshape(-1, size, size, channels)
  w = int(np.sqrt(x.shape[0]))
  img = img.reshape((w, w, size, size, channels)).transpose((0, 2, 1, 3, 4)).reshape((w * size, w * size, channels))
  return img

def visualization(x, name, datasets):
  x = x.permute(0, 2, 3, 1).detach().cpu().numpy()

  img = image_grid(x)
  plt.figure(figsize=(8,8))
  plt.axis('off')
  if datasets =="MNIST":
      plt.imshow(img,cmap='gray')
  else :
      plt.imshow(img)
  plt.savefig(name, dpi= 500)
  plt.show()
def diffusion_animation(samples, name="diffusion_1.8.gif"):
    fig = plt.figure(figsize=(12, 12))
    batch_size = 64
    ims = []
    for i in range(0, len(samples), 10):
        sample = samples[i]
        sample_grid = make_grid(sample, nrow=int(np.sqrt(batch_size)))
        plt.axis('off')
        im = plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
        ims.append([im])

    animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    animate.save(name)
    plt.show()


def sample(path=None, dir_path=None, score_model=None, y=None, alpha=2, beta_min=0.1, beta_max=20,
           num_steps=1000, batch_size=64, LM_steps=1000, sampler='pc_sampler2',
           Predictor=True, Corrector=False, name='', trajectory=False, x_0=False,
           clamp=10, initial_clamp=3, device='cuda', datasets="CelebA", ch=128, ch_mult=[1, 2, 2,2],
           num_res_blocks=2, resolution=64, mode= 'approximation',h=1.5,
           clamp_mode="constant", num_classes=10, conditional=False):
    sde = VPSDE(alpha=alpha, beta_min=beta_min, beta_max=beta_max)

    if datasets == "CIFAR10":
        #score_model = NCSNpp(ch=ch, ch_mult=ch_mult, resolution=resolution, num_res_blocks=num_res_blocks)
        score_model = NCSNpp(ch=ch, ch_mult=ch_mult, resolution=resolution, num_res_blocks=num_res_blocks)

    if datasets == "CIFAR100":
        score_model = Model(ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks)

    if datasets == "CelebA":
        score_model = Model(ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks)

    if conditional == False:
        num_classes = None
    if datasets == "MNIST":
        score_model = Unet(
            dim=28,
            channels=1,
            dim_mults=(1, 2, 4,), num_classes= num_classes)

    score_model = score_model.to(device)
    #score_model = torch.nn.DataParallel(score_model)
    score_model.eval()
    if path:
        ckpt = torch.load(path, map_location=device)
        score_model.load_state_dict(ckpt, strict=False)
        score_model.eval()

    if score_model:
        score_model = score_model

    if sampler == 'pc_sampler2':
        samples = pc_sampler2(score_model,
                              sde=sde, alpha=sde.alpha,
                              batch_size=batch_size, mode= mode,
                              num_steps=num_steps,
                              device=device, x_0=x_0,y=y,
                              Predictor=Predictor, Corrector=Corrector,
                              LM_steps=LM_steps, trajectory=trajectory,
                              clamp=clamp, initial_clamp=initial_clamp, datasets=datasets, clamp_mode=clamp_mode)
    elif sampler == 'dpm_sampler':
        samples = dpm_sampler(score_model,
                              sde=sde, alpha=sde.alpha,
                              batch_size=batch_size,
                              num_steps=num_steps,
                              device=device, x_0=x_0,y=y,
                              Predictor=Predictor, Corrector=Corrector,
                              LM_steps=LM_steps, trajectory=trajectory,
                              clamp=clamp, initial_clamp=initial_clamp, datasets=datasets, clamp_mode=clamp_mode)

    elif sampler == "ode_sampler":
        samples = ode_sampler(score_model,
                              sde=sde, alpha=sde.alpha, h=h,
                              batch_size=batch_size,
                              num_steps=num_steps,
                              device=device,y=y,
                              Predictor=Predictor, Corrector=Corrector, trajectory=trajectory,
                              clamp=clamp, initial_clamp=initial_clamp, datasets=datasets, clamp_mode=clamp_mode)

    if trajectory:
        for i, img in enumerate(samples):
            img = (img + 1) / 2
            img = img.clamp(0.0, 1.0)
            samples[i] = img
        last_sample = samples[-1]
        sample_grid = make_grid(last_sample, nrow=int(np.sqrt(batch_size)))
        plt.figure(figsize=(6, 6))
        plt.axis('off')
        plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    if x_0:
        samples = torch.cat(samples, dim=0)
        samples = (samples + 1) / 2
        samples = samples.clamp(0.0, 1.0)
        sample_grid = make_grid(samples, nrow=int(np.sqrt(batch_size)))
        plt.figure(figsize=(12, 6))
        plt.axis('off')
        plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    else:
        samples = (samples + 1) / 2
        samples = samples.clamp(0.0, 1.0)
        last_sample = samples
        sample_grid = make_grid(last_sample, nrow=int(np.sqrt(batch_size)))
        plt.figure(figsize=(6, 6))
        plt.axis('off')
        plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)

    name = str(name) + str(datasets) + str(
        time.strftime('%m%d_%H%M_', time.localtime(time.time()))) + '_' + 'alpha' + str(f'{alpha}') + 'beta' + str(
        f'{beta_min}') + '_' + str(
        f'{beta_max}') + str(f'{initial_clamp}_{clamp}_{clamp_mode}') + '.png'
    if dir_path:
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
    dir_path = os.path.join(dir_path, 'sample')
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    name = os.path.join(dir_path, name)

    visualization(last_sample, name, datasets)

    plt.show()
    plt.cla()
    plt.clf()
  
    if trajectory:
         name2 = str(datasets)+ str(time.strftime('%m%d_%H%M_', time.localtime(time.time()))) + '_' + 'alpha' + str(
             f'{alpha}') + 'beta' + str(f'{beta_min}') + '_' + str(
             f'{beta_max}') + '.gif'
         name2 = os.path.join(dir_path,name2)
         diffusion_animation(samples, name=name2)

    return samples

    # if trajectory:
    #   name2 = str(datasets)+ str(time.strftime('%m%d_%H%M_', time.localtime(time.time()))) + '_' + 'alpha' + str(f'{