import os
from sampling import *
from torch.optim import Adam
from sample_fid import sample_fid
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
import random
from sample_fid import dataloader2png
from torchvision.datasets import MNIST, CIFAR10, CelebA, CIFAR100
import tqdm
import os
from fid_score import fid_score
from coverage import calculate_given_paths
from scipy.stats import levy_stable
import matplotlib.pyplot as plt
import time
from model import *
from models.ncsnpp import NCSNpp
from losses import *
from classifier import ratio_test
from Diffusion import *
import numpy as np
import torch
from cifar10_model import *
import torchvision.transforms as transforms
from ema import EMAHelper
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize

from transformers import  AdamW, get_scheduler

from torchvision import datasets, transforms


from torchlevy import LevyStable

levy = LevyStable()

image_size = 28
channels = 1
batch_size = 128


def train(alpha=1.9, lr=1e-5, batch_size=128, beta_min=0.1, beta_max=7.5,
          n_epochs=15, num_steps=1000, datasets='MNIST', path=None, device='cuda',
          training_clamp=3, resolution=32, ch=128, ch_mult=[1,2,2,2], num_res_blocks=4,
          dropout=0.1, initial_epoch=0, mode='approximation',total_imbalanced= False,
          imbalanced=False, sample_probs=[0,0.1,0,0,0,0,0,0,0.90,0],
          num_classes = 10, conditional=False, fix_class=1):
    sde = VPSDE(alpha, beta_min=beta_min, beta_max=beta_max, device=device)
    if conditional == False:
        num_classes = None

    if device == 'cuda':
        num_workers = 2
    else:
        num_workers = 4

    if datasets == "MNIST":
        image_size = 28
        channels = 1

        transform = Compose([
            Resize(image_size),
            ToTensor()])
        transformer = transforms.Compose([transforms.Resize((28, 28)),
                                          transforms.ToTensor()],)

        dataset = MNIST('.', train=True, transform=transform, download=True)
        validation_dataset = MNIST(root='./data', train=False, download=True, transform=transformer)


        data_loader = DataLoader(dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers, generator=torch.Generator(device=device))

        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=1, shuffle=False)
        if imbalanced==True:

            idx_to_del = [i for i, label in enumerate(data_loader.dataset.targets) if  random.random() > sample_probs[label]]
            idx_to_del2 = [i for i, label in enumerate(validation_loader.dataset.targets) if random.random() > sample_probs[label]]

            imbalanced_train_dataset = copy.deepcopy(dataset)
            imbalanced_validation_dataset = copy.deepcopy(validation_dataset)

            imbalanced_train_dataset.targets = np.delete(imbalanced_train_dataset.targets, idx_to_del, axis=0)
            imbalanced_validation_dataset.targets= np.delete(imbalanced_validation_dataset.targets, idx_to_del2, axis=0)

            imbalanced_train_dataset.data = np.delete(imbalanced_train_dataset.data, idx_to_del, axis=0)
            imbalanced_validation_dataset.data = np.delete(imbalanced_validation_dataset.data, idx_to_del2, axis=0)

            imbalanced_train_loader = torch.utils.data.DataLoader(imbalanced_train_dataset, batch_size=batch_size,
                                                                  shuffle=True,
                                                                  generator=torch.Generator(device=device))
            imbalanced_validation_loader = torch.utils.data.DataLoader(imbalanced_validation_dataset, batch_size=batch_size,
                                                                  shuffle=True,
                                                                  generator=torch.Generator(device=device))
            data_loader =imbalanced_train_loader
            validation_loader = imbalanced_validation_loader

        score_model = Unet(
            dim=image_size,
            channels=channels,
            dim_mults=(1, 2, 4,), num_classes= num_classes)

    if datasets == "CelebA":
        image_size = 32
        channels = 3

        transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])
        dataset = CelebA('/scratch/private/eunbiyoon/Levy_motion', transform=transform, download=True)

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                 generator=torch.Generator(device=device))
        transformer = transforms.Compose(
            [transforms.Resize((32, 32)), transforms.ToTensor()
             ])
        validation_dataset = CelebA(root='/scratch/private/eunbiyoon/Levy_motion', download=True, transform=transformer)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

        score_model = Model(resolution=image_size, in_channels=channels, out_ch=channels,
                            ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, dropout=dropout)

    if datasets == 'CIFAR100':
        transform = transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])

        dataset = CIFAR100('/scratch/private/eunbiyoon/Levy_motion', train=True, transform=transform, download=True)
        data_loader = DataLoader(dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers, generator=torch.Generator(device=device))
        transformer = transforms.Compose([transforms.Resize((32, 32)),
                                          transforms.ToTensor()
                                          ])
        validation_dataset = CIFAR100(root='/scratch/private/eunbiyoon/Levy_motion', train=False, download=True,
                                     transform=transformer)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

        score_model = Model(ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, dropout=dropout)



    if datasets == 'CIFAR10':
        image_size=32
        channels=3
        transform = transforms.Compose([transforms.Resize((32, 32)),
                                        transforms.RandomHorizontalFlip(p=0.5),
                                        transforms.ToTensor()])

        dataset = CIFAR10('/scratch/private/eunbiyoon/Levy_motion', train=True, transform=transform, download=True)


        data_loader = DataLoader(dataset, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers, generator=torch.Generator(device=device))


        transformer = transforms.Compose([transforms.Resize((32, 32)),
                                          transforms.ToTensor()
                                          ])
        validation_dataset = CIFAR10(root='/scratch/private/eunbiyoon/Levy_motion', train=False, download=True,
                                     transform=transformer)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)
        
        if imbalanced==True:

            idx_to_del = [i for i, label in enumerate(data_loader.dataset.targets) if  random.random() > sample_probs[label]]
            idx_to_del2 = [i for i, label in enumerate(validation_loader.dataset.targets) if random.random() > sample_probs[label]]

            imbalanced_train_dataset = copy.deepcopy(dataset)
            imbalanced_validation_dataset = copy.deepcopy(validation_dataset)

            imbalanced_train_dataset.targets = np.delete(imbalanced_train_dataset.targets, idx_to_del, axis=0)
            imbalanced_validation_dataset.targets= np.delete(imbalanced_validation_dataset.targets, idx_to_del2, axis=0)

            imbalanced_train_dataset.data = np.delete(imbalanced_train_dataset.data, idx_to_del, axis=0)
            imbalanced_validation_dataset.data = np.delete(imbalanced_validation_dataset.data, idx_to_del2, axis=0)

            imbalanced_train_loader = torch.utils.data.DataLoader(imbalanced_train_dataset, batch_size=batch_size,
                                                                  shuffle=True,
                                                                  generator=torch.Generator(device=device))
            imbalanced_validation_loader = torch.utils.data.DataLoader(imbalanced_validation_dataset, batch_size=batch_size,
                                                                  shuffle=True,
                                                                  generator=torch.Generator(device=device))
            data_loader =imbalanced_train_loader
            validation_loader = imbalanced_validation_loader
        score_model = NCSNpp(ch=ch, ch_mult=ch_mult, resolution=32, num_res_blocks=num_res_blocks, dropout=dropout)

    score_model = score_model.to(device)
    #score_model =torch.nn.DataParallel(score_model)
    if path:
        ckpt = torch.load(path, map_location=device)
        score_model.load_state_dict(ckpt, strict=False)

    ema_helper = EMAHelper(mu=0.9999)
    num_training_steps = n_epochs * len(data_loader)
    ema_helper.register(score_model)
    optimizer = AdamW(score_model.parameters(), lr=lr)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    score_model.train()

    counter = 0
    t0 = time.time()
    L = []
    L_val = []


    for epoch in range(n_epochs):
        counter += 1
        avg_loss = 0.
        num_items = 0
        j = 0
        for x, y in data_loader:
            x = 2 * x - 1

            n = x.size(0)
            x = x.to(device)
            y= y.to(device)
            t = torch.rand(x.shape[0]).to(device)*(sde.T-0.00001)+0.00001
            # t = torch.randint(low=0, high=999, size=(n // 2 + 1,)).to(device)
            # t = torch.cat([t, 999 - t - 1], dim=0)[:n]
            # t = (t + 1) / 1000

            #e_L = torch.clamp(levy.sample(alpha, 0, size=x.shape).to(device), -training_clamp,training_clamp)

            #e_L = levy.sample(alpha, 0, size=(x.shape), is_isotropic=True, clamp=20).to(device)
            if alpha==2:
                e_L = torch.randn(size=(x.shape))*np.sqrt(2)
                e_L = e_L.to(device)
            else:
                e_L = levy.sample(alpha, 0, size=(x.shape), is_isotropic=True, clamp=20).to(device)

            if conditional == False:
                y = None
            if np.random.random() < 0.2:
                y = None

            loss = loss_fn(score_model, sde, x, t,y, e_L=e_L, num_steps=num_steps, mode=mode, training_clamp=training_clamp)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(score_model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            ema_helper.update(score_model)


            print(f'{epoch} th epoch {j} th step loss: {loss}')
            j += 1
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

        else:
            with torch.no_grad():
                counter += 1
                val_avg_loss = 0.
                val_num_items = 0
                for x, y in validation_loader:
                    x= 2*x-1
                    n = x.size(0)
                    x = x.to(device)
                    t = torch.rand(x.shape[0]).to(device)*(sde.T-0.00001)+0.00001
                    if conditional == False:
                        y = None
                    if np.random.random() < 0.2:
                        y = None

                    if alpha == 2:
                        e_L = levy.sample(alpha, 0, size=(x.shape), is_isotropic=False).to(device)
                    else:
                        e_L = levy.sample(alpha, 0, size=(x.shape), is_isotropic=True, clamp=20).to(device)

                    val_loss = loss_fn(score_model, sde, x, t,y, e_L, num_steps=num_steps, mode = mode)
                    val_avg_loss += val_loss.item() * x.shape[0]
                    val_num_items += x.shape[0]
        L.append(avg_loss / num_items)
        L_val.append(val_avg_loss / val_num_items)

        t1 = time.time()
        if epoch%5 == 0 :

            name =  mode
            ckpt_name = str(datasets) + str(
            f'batch{batch_size}ch{ch}ch_mult{ch_mult}num_res{num_res_blocks}dropout{dropout}') + str(
            f'clamp{training_clamp}') + str(f'_epoch{epoch+initial_epoch}_') + str(f'{alpha}_{beta_min}_{beta_max}.pth')
            dir_path = str(datasets) + str(f'batch{batch_size}lr{lr}ch{ch}ch_mult{ch_mult}num_res{num_res_blocks}dropout{dropout}') + str(
            f'clamp{training_clamp}') + str(f'{alpha}_{beta_min}_{beta_max}')
            if conditional==True:
                dir_path = 'conditional'+dir_path
            if imbalanced==True:
                dir_path = str(sample_probs)+dir_path
            dir_path = os.path.join('/scratch/private/eunbiyoon/sub_Levy', dir_path)
            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)

            X = range(len(L))
            plt.plot(X, L, 'r', label='training loss')
            plt.plot(X, L_val, 'b', label='validation loss')
            plt.legend()
            name_ = os.path.join(dir_path,  'loss.png')
            plt.savefig(name_)
            plt.cla()

            dir_path2 = os.path.join(dir_path, 'ckpt')
            if not os.path.isdir(dir_path2):
                os.mkdir(dir_path2)
            if conditional==False:
                fix_class = None

            ckpt_name = os.path.join(dir_path2, ckpt_name)
            torch.save(score_model.state_dict(), ckpt_name)
            name= str(epoch+initial_epoch)+mode
            sample(alpha=sde.alpha, path=ckpt_name,
               beta_min=beta_min, beta_max=beta_max, sampler='pc_sampler2', batch_size=64, num_steps=num_steps, LM_steps=50,
               Predictor=True, Corrector=False, trajectory=False, clamp=training_clamp, initial_clamp=training_clamp,
               clamp_mode="constant",
               datasets=datasets, name=name,conditional=conditional, y= fix_class,
               dir_path=dir_path, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, resolution=resolution)
        # name = 'ode'+ str(epoch + initial_epoch) + mode
        # sample(alpha=sde.alpha, path=ckpt_name,
        #        beta_min=beta_min, beta_max=beta_max, sampler='ode_sampler', batch_size=64, num_steps=20,
        #        LM_steps=50,
        #        Predictor=True, Corrector=False, trajectory=False, clamp=2.3, initial_clamp=training_clamp,
        #        clamp_mode="constant",
        #        datasets=datasets, name=name,
        #        dir_path=dir_path, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, resolution=resolution )
    if imbalanced==True:
        s = str(sample_probs)
        ratio_test(path=ckpt_name,conditional=conditional,num_classes=num_classes,
               alpha=alpha, datasets=datasets, device=device,
               ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, resolution=resolution, name=s)

    dir_path = str(datasets) + str(
        f'batch{batch_size}lr{lr}ch{ch}ch_mult{ch_mult}num_res{num_res_blocks}dropout{dropout}') + str(
        f'clamp{training_clamp}') + str(f'{alpha}_{beta_min}_{beta_max}')
    if conditional == True:
        dir_path = 'conditional' + dir_path
    if imbalanced == True:
        dir_path = str(sample_probs) + dir_path

    dir_path = os.path.join('/scratch/private/eunbiyoon/sub_Levy', dir_path)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
        
    name = datasets+'_'+str(alpha)
    name2 = 'com'+datasets
    if conditional == False:
        fix_class= None
    if imbalanced:
        name = str(sample_probs) + name
        name2 = str(sample_probs) + name2
    if fix_class:
        name = str(fix_class)+name
        name2 = str(fix_class)+name2
    image_folder =os.path.join(dir_path  , name)
    com_folder = os.path.join(dir_path  , name2)

    if n_epochs==0:
        ckpt_name = path
    if not os.path.isdir(image_folder):
        os.mkdir(image_folder)
    if not os.path.isdir(com_folder):
        os.mkdir(com_folder)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=1,shuffle=False)

        dataloader2png(validation_loader,com_folder,datasets)

    plt.axis('off')
    sample_fid(
        path=ckpt_name,
        image_folder=image_folder, mode='approximation',y=fix_class,
        alpha=alpha, beta_min=0.1, beta_max=20, total_n_samples=10000,
        num_steps=1000,
        channels=channels,
        image_size=image_size, batch_size=500, device='cuda', datasets=datasets,
        ch=ch, ch_mult=ch_mult, resolution=resolution,
        sampler="pc_sampler2",
        num_res_blocks=num_res_blocks, conditional=conditional
        )
    # if datasets == "MNIST":
    #     fid= fid_score('/scratch/private/eunbiyoon/sub_Levy/mnist',image_folder )
    #     print(f'FID{fid}')
    #     metrics = calculate_given_paths('/scratch/private/eunbiyoon/sub_Levy/mnist', image_folder)
    #     print('coverages', metrics)
    # elif datasets == "CIFAR10":
    #     fid= fid_score('/scratch/private/eunbiyoon/sub_Levy/cifar10',image_folder )
    #     print(f'FID{fid}')
    #     metrics = calculate_given_paths('/scratch/private/eunbiyoon/sub_Levy/cifar10', image_folder)
    #     print('coverages', metrics)
    fid = fid_score(com_folder, image_folder)
    print(f'FID{fid}')
    metrics = calculate_given_paths(com_folder, image_folder)
    print('coverages', metrics)

        