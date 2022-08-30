import os


from model import *
from Diffusion import *
from sampler import *
import torch
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

image_size = 28
channels = 1
batch_size = 128


def sample(path='ckpt.pth', alpha=2,beta_min=0.1, beta_max=20,
           num_steps = 1000, batch_size = 64, LM_steps=1000, sampler ='pc_sampler2',
           Predictor=True, Corrector=False, name='image', trajectory=False):

    sde = VPSDE(alpha=alpha,beta_min=beta_min, beta_max=beta_max)
    sde = VPSDE(alpha, beta_min = beta_min, beta_max = beta_max)
    score_model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,))

    score_model = score_model.to(device)
    ckpt = torch.load(path, map_location=device)
    score_model.load_state_dict(ckpt,  strict=False)


    if sampler =='pc_sampler2':
        samples = pc_sampler2(score_model,
                          sde=sde, alpha=sde.alpha,
                          batch_size=batch_size,
                          num_steps=num_steps,
                          device=device,
                          Predictor=Predictor, Corrector=Corrector,
                          LM_steps = LM_steps, trajectory=trajectory)
    elif sampler == 'dpm_sampler':
        samples = dpm_sampler(score_model,
                          sde=sde, alpha=sde.alpha,
                          batch_size=batch_size,
                          num_steps=num_steps,
                          device=device)
    if trajectory:
        for i, img in enumerate(samples):
            img = (img+1)/2
            img = img.clamp(0.0, 1.0)
            samples[i]= img
        last_sample = samples[-1]
    else:
        samples = (samples+1)/2
        samples = samples.clamp(0.0, 1.0)
        last_sample = samples
    sample_grid = make_grid(last_sample, nrow=int(np.sqrt(batch_size)))
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)

    name = str(time.strftime('%m%d_%H%M_', time.localtime(time.time()))) + '_'+ 'alpha'+str(f'{alpha}')+'beta'+ str(f'{beta_min}') + '_'+str(
        f'{beta_max}') + '.png'

    dir_path = os.path.join(os.getcwd(), 'sample')
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    name = os.path.join(dir_path, name)
    plt.savefig(name)
    plt.show()
    return samples
