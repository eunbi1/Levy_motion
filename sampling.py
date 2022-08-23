import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="5"

from model import *
from Diffusion import *
from sampler import *
import torch
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def sample(path='ckpt.pth', alpha=2,beta_min=0.1, beta_max=20,
           num_steps = 1000, batch_size = 64, sampler ='pc_sampler2',
           Predictor=True, Corrector=False, name='image' ):

    sde = VPSDE(alpha=alpha,beta_min=beta_min, beta_max=beta_max)
    score_model = ScoreNet(sde)
    score_model = score_model.to(device)
    ckpt = torch.load(path, map_location=device)
    score_model.load_state_dict(ckpt,  strict=False)

    if sampler =='pc_sampler2':
        samples = pc_sampler2(score_model,
                          sde=sde, alpha=sde.alpha,
                          batch_size=batch_size,
                          num_steps=num_steps,
                          device=device,
                          Predictor=True, Corrector=False)
    elif sampler == 'dpm_sampler':
        samples = dpm_sampler(score_model,
                          sde=sde, alpha=sde.alpha,
                          batch_size=batch_size,
                          num_steps=num_steps,
                          device=device)

    samples = samples.clamp(0.0, 1.0)
    sample_grid = make_grid(samples, nrow=int(np.sqrt(batch_size)))
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    dir_path= os.path.join(os.getcwd(), 'sample')
    name = str(time.strftime('%m%d_%H%M', time.localtime(time.time()))) + str(f'{beta_min}') + str(
            f'{beta_max}') + '.png'
    if not os.path.isdir(dir_path):
      os.mkdir(dir_path)
    name=os.path.join(dir_path,name)
    plt.savefig(name)
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    plt.show()
