

from model import *
from Diffusion import *
from sampler import *
import torch
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

## Load the pre-trained checkpoint from disk.



#@title Sampling

num_steps =  500
alpha=2
batch_size = 64


score_model = torch.nn.DataParallel(ScoreNet())
score_model = score_model.to(device)
ckpt = torch.load('ckpt.pth', map_location=device)
score_model.load_state_dict(ckpt)



#model, sde definition
sample_batch_size = 64

sde =VPSDE(alpha=alpha)

sampler ="pc_sampler2"#@param ['dpm_sampler', 'pc_sampler', 'pc_sampler2', 'nothing']{'type' : 'string'}
## Generate samples using the specified sampler.
samples = pc_sampler2(score_model,
                  sde,
                  alpha=alpha,
                  batch_size=batch_size,
                  num_steps=num_steps,
                  device=device, Corrector = False, Predictor=True)


samples = samples.clamp(0.0, 1.0)
sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))
plt.figure(figsize=(6,6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)




