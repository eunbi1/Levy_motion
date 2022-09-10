from torchvision.utils import make_grid
from torchvision.datasets import MNIST, CIFAR10, CelebA, CIFAR100
import tqdm
from scipy.stats import levy_stable
import matplotlib.pyplot as plt
from losses import *
import numpy as np
import torch
from Diffusion import *
import random
import torch.backends.cudnn as cudnn
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize





from torchlevy import LevyStable
levy = LevyStable()



## Sample visualization.

def visualization(samples, sample_batch_size=64):
    samples = samples.clamp(0.0, 1.0)
    sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
    plt.show()


def gamma_func(x):
    return torch.tensor(gamma(x))


def get_discrete_time(t, N=1000):
    return N * t


def ddim_score_update2(score_model, sde, x_s, s, t, h=0.6, clamp = 10, device='cuda'):
    score_s = score_model(x_s, s) * torch.pow(sde.marginal_std(s) + 1e-4, -1)[:, None, None, None]
    time_step = s-t
    beta_step = sde.beta(s)*time_step
    x_coeff = 1 + beta_step/sde.alpha
    if sde.alpha==2:
        score_coeff2 = torch.pow(beta_step, 2 / sde.alpha) * gamma_func(sde.alpha + 1)

    else:
        score_coeff2 = torch.pow(beta_step, 2/sde.alpha)*torch.pow(time_step, 1-2/sde.alpha)*np.sin(torch.pi/2*(2-sde.alpha))/(2-sde.alpha)*2/torch.pi*gamma_func(sde.alpha+1)
    noise_coeff = torch.pow(beta_step, 1 / sde.alpha)

    e_L = torch.clamp(levy.sample(sde.alpha, 0, size=x_s.shape ).to(device),-clamp,clamp)

    x_t = x_coeff[:, None, None, None] * x_s + score_coeff2[:, None, None, None] * score_s + noise_coeff[:, None, None,None] * e_L
    #print('score_coee', torch.min(score_coeff), torch.max(score_coeff))
    #print('noise_coeff',torch.min(noise_coeff), torch.max(noise_coeff))
    #print('x_coeff', torch.min(x_coeff), torch.max(x_coeff))
    """
    print('x_s range', torch.min(x_s), torch.max(x_s))
    print('x_t range', torch.min(x_t), torch.max(x_t))
    print('x_s mean', torch.mean(x_s))
    print('x_t mean', torch.mean(x_t))
    print('score range',torch.min(score_s), torch.max(score_s))
    """


    #print('x coeff adding', torch.min(x_coeff[:, None, None, None] * x_s), torch.max(x_coeff[:, None, None, None] * x_s))
    #print('score adding',torch.min(score_coeff[:, None, None, None] * score_s), torch.max(score_coeff[:, None, None, None] * score_s) )
    #print('noise adding', torch.min(noise_coeff[:, None, None,None] * e_L), torch.max(noise_coeff[:, None, None,None] * e_L))

    return x_t


def pc_sampler2(score_model,
                sde,
                alpha,
                batch_size,
                num_steps,
                LM_steps=200,
                device='cuda',
                eps=1e-4,
                x_0= False,
                Predictor=True,
                Corrector=False, trajectory=False,
                clamp = 10,
                initial_clamp =3, final_clamp = 1,
                datasets="MNIST", clamp_mode = 'constant'):


    t = torch.ones(batch_size, device=device)

    if datasets =="MNIST":
        if x_0:
            transformer = transforms.Compose([transforms.Resize((28, 28)),
                                              transforms.ToTensor()])
            validation_dataset = MNIST(root='./data', train=False, download=True, transform=transformer)
            validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
            x,y = next(iter(validation_loader))
            e_L = levy.sample(alpha, 0, (batch_size, 1, 28, 28)).to(device)
            e_L = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]
        else:
         e_L = levy.sample(alpha, 0, (batch_size, 1, 28,28)).to(device)
         x_s = torch.clamp(e_L,-initial_clamp,initial_clamp) *sde.marginal_std(t)[:,None,None,None]

    elif datasets =="CIFAR10":
        if x_0:
            transformer = transforms.Compose([transforms.Resize((32, 32)),
                                              transforms.ToTensor()
                                              ])
            validation_dataset = CIFAR10(root='/scratch/private/eunbiyoon/data', train=False, download=True,
                                         transform=transformer)
            validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
            x,y = next((iter(validation_loader)))
            e_L = levy.sample(alpha, 0, (batch_size, 3, 32, 32)).to(device)
            e_L = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]


        else:
         e_L = levy.sample(alpha, 0, (batch_size, 3, 32, 32)).to(device)
         x_s = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]

            
    elif datasets == "CIFAR100":
        if x_0:
            transformer = transforms.Compose([transforms.Resize((32, 32)),
                                              transforms.ToTensor()
                                              ])
            validation_dataset = CIFAR100(root='/scratch/private/eunbiyoon/data', train=False, download=True,
                                         transform=transformer)
            validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
            x,y = next((iter(validation_loader)))
            e_L = levy.sample(alpha, 0, (batch_size, 3, 32, 32)).to(device)
            e_L = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]
        else:
         e_L = levy.sample(alpha, 0, (batch_size, 3, 32,32)).to(device)
         x_s = torch.clamp(e_L,-initial_clamp,initial_clamp) *sde.marginal_std(t)[:,None,None,None]

    elif datasets =="CelebA":
        if x_0:
            transformer = transforms.Compose(
                [transforms.Resize((32, 32)), transforms.ToTensor()
                 ])
            validation_dataset = CelebA(root='/scratch/private/eunbiyoon/data', download=True, transform=transformer)
            validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
            x,y = next(iter(validation_loader))
            e_L = levy.sample(alpha, 0, (batch_size, 3, 32, 32)).to(device)
            e_L = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]
        else:
         e_L = levy.sample(alpha, 0, (batch_size, 3, 32,32)).to(device)
         x_s = torch.clamp(e_L,-initial_clamp,initial_clamp) *sde.marginal_std(t)[:,None,None,None]

    if x_0 :
        sigma = sde.marginal_std(t)
        x_coeff = sde.diffusion_coeff(t)
        x = 2*x-1
        x = x.to(device)
        x_s = x_coeff[:, None, None, None] * x + e_L * sigma[:, None, None, None]

        
    if trajectory:
        samples = []
        samples.append(x_s)

    time_steps = torch.linspace(1-1e-4, 0, num_steps)
    step_size = time_steps[0] - time_steps[1]

    batch_time_step_s = torch.ones(x_s.shape[0])
    batch_time_step_s = batch_time_step_s.to(device)

    with torch.no_grad():
        for t in tqdm.tqdm(time_steps):
            batch_time_step_t = torch.ones(x_s.shape[0])*t
            batch_time_step_t = batch_time_step_t.to(device)

            if clamp_mode == "constant":
                linear_clamp = clamp
            if clamp_mode == "linear":
                linear_clamp = batch_time_step_t[0]*(clamp-final_clamp)+final_clamp
            if clamp_mode == "root":
                linear_clamp = torch.pow(batch_time_step_t[0],1/2)*(clamp-final_clamp)+final_clamp
            if clamp_mode == "quad":
                linear_clamp = batch_time_step_t[0]**2*(clamp-final_clamp)+final_clamp
            #linear_clamp = torch.pow(batch_time_step_t[0], 1 / 2) * (5 - 1) + 1
            if Predictor:
                x_s = ddim_score_update2(score_model, sde, x_s, batch_time_step_s, batch_time_step_t, clamp = linear_clamp)
            if Corrector:
                for j in range(LM_steps):
                    grad = score_model(x_s, batch_time_step_t) * torch.pow(sde.marginal_std(batch_time_step_t)+1e-5, -1)[:, None, None, None]
                    if datasets == "MNIST":
                        e_L = levy.sample(alpha, 0, (batch_size, 1, 28, 28)).to(device)
                        e_L = torch.clamp(e_L, -clamp, clamp)
                    elif datasets == "CIFAR10":
                        e_L = levy.sample(alpha, 0, (batch_size, 3, 32, 32)).to(device)
                        e_L = torch.clamp(e_L, -clamp, clamp)

                    elif datasets == "CelebA":
                        e_L = levy.sample(alpha, 0, (batch_size, 3, 32, 32)).to(device)
                        e_L = torch.clamp(e_L, -clamp, clamp)

                    x_s = x_s + step_size * gamma_func(sde.alpha - 1) / (
                            gamma_func(sde.alpha / 2) ** 2) * grad + torch.pow(
                        step_size, 1 / sde.alpha) * e_L
            if trajectory:
                samples.append(x_s)
            batch_time_step_s = batch_time_step_t




    if trajectory:
        return samples
    if x_0:
        return x, x_s
    else:
        return x_s


def ode_score_update(score_model, sde, x_s, s, t, clamp=3, h=0.001, return_noise=False):
    score_s = score_model(x_s, s) * torch.pow(sde.marginal_std(s) + 1e-4, -1)[:, None, None, None]
    time_step = s - t
    beta_step = sde.beta(s) * time_step
    x_coeff = 1 + beta_step / sde.alpha
    #x_coeff = sde.diffusion_coeff(t)*torch.pow(sde.diffusion_coeff(s), -1)

    if sde.alpha==2:
        score_coeff = 1/2*torch.pow(beta_step, 2 / sde.alpha) * gamma_func(sde.alpha + 1)

    else:
        score_coeff = 1/2*torch.pow(beta_step, 2/sde.alpha)*torch.pow(time_step, 1-2/sde.alpha)*np.sin(torch.pi/2*(2-sde.alpha))/(2-sde.alpha)*2/torch.pi*gamma_func(sde.alpha+1)
        score_coeff = 1/2*sde.diffusion_coeff(t)*torch.pow(sde.diffusion_coeff(s), -1)*torch.pow(beta_step, 2/sde.alpha)*torch.pow(time_step, 1-2/sde.alpha)*np.sin(torch.pi/2*(2-sde.alpha))/(2-sde.alpha)*2/torch.pi*gamma_func(sde.alpha+1)
        #score_coeff =  sde.diffusion_coeff(t)*torch.pow(sde.diffusion_coeff(s), -1)*beta_step*gamma_func(sde.alpha-1)/gamma_func(sde.alpha/2)**2/np.power(h, sde.alpha-2)
    x_t = x_coeff[:, None, None, None] * x_s + score_coeff[:, None, None, None] * score_s

    return x_t

def ode_sampler(score_model,
                    sde,
                    alpha,
                    batch_size,
                    num_steps,
                    LM_steps=200,
                    device='cuda',
                    eps=1e-4,
                    x_0=False,
                    Predictor=True,
                    Corrector=False, trajectory=False,
                    clamp=10,
                    initial_clamp=3, final_clamp=1,
                    datasets="MNIST", clamp_mode='constant'):

    t = torch.ones(batch_size, device=device)

    if datasets == "MNIST":
            if x_0:
                transformer = transforms.Compose([transforms.Resize((28, 28)),transforms.ToTensor()])
                validation_dataset = MNIST(root='./data', train=False, download=True, transform=transformer)
                validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size,
                                                                shuffle=False)
                x, y = next(iter(validation_loader))
                e_L = levy.sample(alpha, 0, (batch_size, 1, 28, 28)).to(device)
                e_L = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]
            else:
                e_L = levy.sample(alpha, 0, (batch_size, 1, 28, 28)).to(device)
                x_s = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]

    elif datasets == "CIFAR10":
            if x_0:
                transformer = transforms.Compose([transforms.Resize((32, 32)),
                                                  transforms.ToTensor()
                                                  ])
                validation_dataset = CIFAR10(root='/scratch/private/eunbiyoon/data', train=False, download=True,
                                             transform=transformer)
                validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size,
                                                                shuffle=False)
                x, y = next((iter(validation_loader)))
                e_L = levy.sample(alpha, 0, (batch_size, 3, 32, 32)).to(device)
                e_L = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]
            else:
                e_L = levy.sample(alpha, 0, (batch_size, 3, 32, 32)).to(device)
                x_s = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]

    elif datasets == "CIFAR100":
            if x_0:
                transformer = transforms.Compose([transforms.Resize((32, 32)),
                                                  transforms.ToTensor()
                                                  ])
                validation_dataset = CIFAR100(root='/scratch/private/eunbiyoon/data', train=False, download=True,
                                              transform=transformer)
                validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size,
                                                                shuffle=False)
                x, y = next((iter(validation_loader)))
                e_L = levy.sample(alpha, 0, (batch_size, 3, 32, 32)).to(device)
                e_L = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]
            else:
                e_L = levy.sample(alpha, 0, (batch_size, 3, 32, 32)).to(device)
                x_s = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]

    elif datasets == "CelebA":
            if x_0:
                transformer = transforms.Compose(
                    [transforms.Resize((32, 32)), transforms.ToTensor()
                     ])
                validation_dataset = CelebA(root='/scratch/private/eunbiyoon/data', download=True,
                                            transform=transformer)
                validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size,
                                                                shuffle=False)
                x, y = next(iter(validation_loader))
                e_L = levy.sample(alpha, 0, (batch_size, 3, 32, 32)).to(device)
                e_L = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]
            else:
                e_L = levy.sample(alpha, 0, (batch_size, 3, 32, 32)).to(device)
                x_s = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]

    if x_0:
            sigma = sde.marginal_std(t)
            x_coeff = sde.diffusion_coeff(t)
            x = 2 * x - 1
            x = x.to(device)
            x_s = x_coeff[:, None, None, None] * x + e_L * sigma[:, None, None, None]

    t = torch.ones(batch_size, device=device)
    if datasets == "MNIST":
        e_L = levy.sample(alpha, 0, (batch_size, 1, 28, 28)).to(device)
        x_s = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]
    elif datasets == "CIFAR10":
        e_L = levy.sample(alpha, 0, (batch_size, 3, 32, 32)).to(device)
        x_s = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]

    elif datasets == "CelebA":
        e_L = levy.sample(alpha, 0, (batch_size, 3, 32, 32)).to(device)
        x_s = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]
    if x_0:
        x_s = x_0

    if trajectory:
        samples = []
        samples.append(x_s)
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]

    batch_time_step_s = torch.ones(x_s.shape[0]) * time_steps[0]
    batch_time_step_s = batch_time_step_s.to(device)

    with torch.no_grad():
        for t in tqdm.tqdm(time_steps[1:]):
            batch_time_step_t = torch.ones(x_s.shape[0]) * t
            batch_time_step_t = batch_time_step_t.to(device)

            if clamp_mode == "constant":
                linear_clamp = clamp
            if clamp_mode == "linear":
                linear_clamp = batch_time_step_t[0] * (clamp - final_clamp) + final_clamp
            if clamp_mode == "root":
                linear_clamp = torch.pow(batch_time_step_t[0], 1 / 2) * (clamp - final_clamp) + final_clamp
            if clamp_mode == "quad":
                linear_clamp = batch_time_step_t[0] ** 2 * (clamp - final_clamp) + final_clamp


            x_s =ode_score_update(score_model, sde, x_s, batch_time_step_s, batch_time_step_t,
                                         clamp=linear_clamp)
            if trajectory:
                samples.append(x_s)
            batch_time_step_s = batch_time_step_t


    if trajectory:
     return samples
    if x_0:
     return x, x_s
    else:
     return x_s



def dpm_score_update(model, sde, x_s, s, t, alpha, h=0.06, return_noise=False):
    log_alpha_s, log_alpha_t = sde.marginal_log_mean_coeff(s), sde.marginal_log_mean_coeff(t)
    lambda_s = sde.marginal_lambda(s)
    lambda_t = sde.marginal_lambda(t)
    sigma_s = sde.marginal_std(s)
    sigma_t = sde.marginal_std(t)
    time_step = s-t
    beta_step = sde.beta(s)*time_step

    score_s = model(x_s, s)*torch.pow( sde.marginal_std(s) + 1e-5, -1)[:,None,None,None]

    h_t = lambda_t - lambda_s

    x_coeff = torch.exp(log_alpha_t - log_alpha_s)

    score_coeff = sigma_t * torch.pow(sigma_s, alpha - 1) * alpha * torch.expm1(h_t) \
                  * gamma_func(alpha - 1) / torch.pow(gamma_func(alpha / 2), 2) / np.power(h, alpha - 2)
    x_t = x_coeff[:, None, None, None] * x_s - score_coeff[:, None, None, None] * score_s

    return x_t



def dpm_sampler(score_model,
                    sde,
                    alpha,
                    batch_size,
                    num_steps,
                    LM_steps=200,
                    device='cuda',
                    eps=1e-4,
                    x_0=False,
                    Predictor=True,
                    Corrector=False, trajectory=False,
                    clamp=10,
                    initial_clamp=3, final_clamp=1,
                    datasets="MNIST", clamp_mode='constant'):

    t = torch.ones(batch_size, device=device)

    if datasets == "MNIST":
            if x_0:
                transformer = transforms.Compose([transforms.Resize((28, 28)),transforms.ToTensor()])
                validation_dataset = MNIST(root='./data', train=False, download=True, transform=transformer)
                validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size,
                                                                shuffle=False)
                x, y = next(iter(validation_loader))
                e_L = levy.sample(alpha, 0, (batch_size, 1, 28, 28)).to(device)
                e_L = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]
            else:
                e_L = levy.sample(alpha, 0, (batch_size, 1, 28, 28)).to(device)
                x_s = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]

    elif datasets == "CIFAR10":
            if x_0:
                transformer = transforms.Compose([transforms.Resize((32, 32)),
                                                  transforms.ToTensor()
                                                  ])
                validation_dataset = CIFAR10(root='/scratch/private/eunbiyoon/data', train=False, download=True,
                                             transform=transformer)
                validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size,
                                                                shuffle=False)
                x, y = next((iter(validation_loader)))
                e_L = levy.sample(alpha, 0, (batch_size, 3, 32, 32)).to(device)
                e_L = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]
            else:
                e_L = levy.sample(alpha, 0, (batch_size, 3, 32, 32)).to(device)
                x_s = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]

    elif datasets == "CIFAR100":
            if x_0:
                transformer = transforms.Compose([transforms.Resize((32, 32)),
                                                  transforms.ToTensor()
                                                  ])
                validation_dataset = CIFAR100(root='/scratch/private/eunbiyoon/data', train=False, download=True,
                                              transform=transformer)
                validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size,
                                                                shuffle=False)
                x, y = next((iter(validation_loader)))
                e_L = levy.sample(alpha, 0, (batch_size, 3, 32, 32)).to(device)
                e_L = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]
            else:
                e_L = levy.sample(alpha, 0, (batch_size, 3, 32, 32)).to(device)
                x_s = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]

    elif datasets == "CelebA":
            if x_0:
                transformer = transforms.Compose(
                    [transforms.Resize((32, 32)), transforms.ToTensor()
                     ])
                validation_dataset = CelebA(root='/scratch/private/eunbiyoon/data', download=True,
                                            transform=transformer)
                validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size,
                                                                shuffle=False)
                x, y = next(iter(validation_loader))
                e_L = levy.sample(alpha, 0, (batch_size, 3, 32, 32)).to(device)
                e_L = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]
            else:
                e_L = levy.sample(alpha, 0, (batch_size, 3, 32, 32)).to(device)
                x_s = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]

    if x_0:
            sigma = sde.marginal_std(t)
            x_coeff = sde.diffusion_coeff(t)
            x = 2 * x - 1
            x = x.to(device)
            x_s = x_coeff[:, None, None, None] * x + e_L * sigma[:, None, None, None]

    t = torch.ones(batch_size, device=device)
    if datasets == "MNIST":
        e_L = levy.sample(alpha, 0, (batch_size, 1, 28, 28)).to(device)
        x_s = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]
    elif datasets == "CIFAR10":
        e_L = levy.sample(alpha, 0, (batch_size, 3, 32, 32)).to(device)
        x_s = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]

    elif datasets == "CelebA":
        e_L = levy.sample(alpha, 0, (batch_size, 3, 32, 32)).to(device)
        x_s = torch.clamp(e_L, -initial_clamp, initial_clamp) * sde.marginal_std(t)[:, None, None, None]
    if x_0:
        x_s = x_0

    if trajectory:
        samples = []
        samples.append(x_s)
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]

    batch_time_step_s = torch.ones(x_s.shape[0]) * time_steps[0]
    batch_time_step_s = batch_time_step_s.to(device)

    with torch.no_grad():
        for t in tqdm.tqdm(time_steps[1:]):
            batch_time_step_t = torch.ones(x_s.shape[0]) * t
            batch_time_step_t = batch_time_step_t.to(device)

            if clamp_mode == "constant":
                linear_clamp = clamp
            if clamp_mode == "linear":
                linear_clamp = batch_time_step_t[0] * (clamp - final_clamp) + final_clamp
            if clamp_mode == "root":
                linear_clamp = torch.pow(batch_time_step_t[0], 1 / 2) * (clamp - final_clamp) + final_clamp
            if clamp_mode == "quad":
                linear_clamp = batch_time_step_t[0] ** 2 * (clamp - final_clamp) + final_clamp


            x_s =dpm_score_update(score_model, sde, x_s, batch_time_step_s , batch_time_step_t, alpha=sde.alpha, h=0.06, return_noise=False)
            if trajectory:
                samples.append(x_s)
            batch_time_step_s = batch_time_step_t


    if trajectory:
     return samples
    if x_0:
     return x, x_s
    else:
     return x_s