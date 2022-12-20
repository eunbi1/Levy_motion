import os

import os
import torch
import tqdm
from tqdm.asyncio import trange, tqdm
from sampler import pc_sampler2, ode_sampler
from torchlevy import LevyStable
from fid_score import fid_score
from coverage import calculate_given_paths
from models.ncsnpp import NCSNpp

levy = LevyStable()
import torchvision.utils as tvu
from Diffusion import VPSDE
from cifar10_model import Model
import glob
import cv2
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST, CIFAR10, CelebA, CIFAR100
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from model import *


def testimg(path, png_name='test_tile'):
    path = path + "/*"
    # path = "./levy-only/*"
    file_list = glob.glob(path)

    images = []

    for file in file_list:
        img = cv2.imread(file)
        images.append(img)

    tile_size = 8

    # 바둑판 타일형 만들기
    def img_tile(img_list):
        return cv2.vconcat([cv2.hconcat(img) for img in img_list])

    imgs = []

    for i in range(0, len(images), tile_size):
        assert tile_size**2 == len(images), "invalid tile size"
        imgs.append(images[i:i+tile_size])

    img_tile_res = img_tile(imgs)
    png_name = "./" + png_name + ".png"
    cv2.imwrite(png_name, img_tile_res)


def sample_fid(path="/scratch/private/eunbiyoon/sub_Levy/[0, 0.1, 0, 0, 0, 0, 0, 0, 0.9, 0]conditionalMNISTbatch128lr0.0001ch128ch_mult[1, 2, 2, 2]num_res4dropout0.1clamp202.0_0.1_20/ckpt/MNISTbatch128ch128ch_mult[1, 2, 2, 2]num_res4dropout0.1clamp20_epoch1405_2.0_0.1_20.pth",
               image_folder='/scratch/private/eunbiyoon/sub_Levy/mnist_20_1',mode= 'approximation',
               alpha=2.0, beta_min=0.1, beta_max=20,total_n_samples= 1000,
               num_steps=1000,num_classes=None,
               channels=3,
               image_size=64, batch_size=100, device='cuda', datasets= 'MNIST',
               ch=128, ch_mult=[1, 2, 2, 2],resolution=32,
               sampler = "pc_sampler2",
               num_res_blocks=2,conditional=False,
               y=None
               ):
    if not os.path.isdir(image_folder):
        os.mkdir(image_folder)

    total_n_samples = total_n_samples # total num of datasamples (cifar10 has 50000 training dataset)
    n_rounds = total_n_samples  // batch_size

    sde = VPSDE(alpha=alpha, beta_min=beta_min, beta_max=beta_max)
    if conditional == None:
        num_classes=None
    if datasets == "CIFAR10":
        score_model = NCSNpp(ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks, resolution=32, num_classes=num_classes)

    if datasets == "CIFAR100":
        score_model = Model(ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks)

    if datasets == "CelebA":
        score_model = Model(ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks)

    if datasets == "MNIST":
        score_model = Unet(
            dim=28,
            channels=1,
            dim_mults=(1, 2, 4,), num_classes=num_classes)

    score_model = score_model.to(device)
    #score_model = torch.nn.DataParallel(score_model)
    if path:
        ckpt = torch.load(path, map_location=device)
        score_model.load_state_dict(ckpt, strict=True)
        score_model.eval()

    j=0
    with torch.no_grad():
        for _ in trange(n_rounds, desc="Generating image samples for FID evaluation."):
            n = batch_size

            x_shape = (n, channels, image_size, image_size)
            x = levy.sample(sde.alpha, 0, size=x_shape).to(device)

            if sde.alpha < 2.0:
                t = torch.ones(n, device=device)
                x = x * sde.marginal_std(t)[:, None, None, None]
            if sampler =="ode_sampler":
             x = ode_sampler(score_model,
                sde,
                sde.alpha,
                batch_size,
                num_steps=20,
                h=1.5,
                device='cuda',
                clamp = 20,
                initial_clamp =20, final_clamp = 10,
                datasets=datasets, clamp_mode = 'constant')
            if sampler =="pc_sampler2":
                x = pc_sampler2(score_model,
                                sde,
                                sde.alpha,
                                batch_size,
                                num_steps=num_steps,
                        

                                device='cuda',mode= mode,


                                clamp=20,
                                initial_clamp=20, final_clamp=3,
                                datasets=datasets, clamp_mode='constant', y= y)

            x = (x+1)/2
            x = x.clamp(0.0, 1.0)

            for i in range(n):
                sam = x[i]

                plt.axis('off')
                if datasets == 'MNIST':
                    fig = plt.figure(figsize=(1,1))
                    fig.patch.set_visible(False)
                    ax = fig.add_subplot(111)
                    ax.set_axis_off()
                    ax.imshow(sam.permute(1, 2, 0).cpu(), vmin=0., vmax=1., cmap='gray')
                else:
                    fig = plt.figure()
                    fig.patch.set_visible(False)
                    ax = fig.add_subplot(111)
                    ax.set_axis_off()
                    ax.imshow(sam.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
                    
                   
                name = str(f'{j}') + '.png'
                name = os.path.join(image_folder, name)

                plt.savefig(name)
                #plt.savefig(name, dpi=500)
                plt.cla()
                plt.clf()
                j= j+1
    if datasets =="MNIST":
        fid= fid_score("/scratch/private/eunbiyoon/sub_Levy/mnist", image_folder)
        print(f'FID{fid}')
        metrics = calculate_given_paths("/scratch/private/eunbiyoon/sub_Levy/mnist",image_folder)
        print('coverages', metrics)
    elif datasets =="CIFAR10":
        fid= fid_score("/scratch/private/eunbiyoon/sub_Levy/cifar10", image_folder)
        print(f'FID{fid}')
        metrics = calculate_given_paths("/scratch/private/eunbiyoon/sub_Levy/cifar10",image_folder)
        print('coverages', metrics)




def dataloader2png(data_loader, com_folder,datasets):
    j=0
    for x,y in tqdm(data_loader):
        x = x.to('cuda')
        n = len(x)
        for i in range(n):
            sam = x[i]
            plt.axis('off')
            if datasets == 'MNIST':
                fig = plt.figure(figsize=(1, 1))
                fig.patch.set_visible(False)
                ax = fig.add_subplot(111)
                ax.set_axis_off()
                ax.imshow(sam.permute(1, 2, 0).cpu(), vmin=0., vmax=1., cmap='gray')
            else:
                fig = plt.figure()
                fig.patch.set_visible(False)
                ax = fig.add_subplot(111)
                ax.set_axis_off()
                ax.imshow(sam.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
            name = str(f'{j}') + '.png'
            name = os.path.join(com_folder, name)
            plt.savefig(name)
            plt.cla()
            plt.clf()
            j = j + 1
def cifar102png(path='/scratch/private/eunbiyoon/sub_Levy/cifar10_1'):
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                      transforms.ToTensor()
                                      ])
    dataset = CIFAR10('/home/eunbiyoon/comb_Levy_motion', train=False, transform=transform, download=True)
    data_loader = DataLoader(dataset, batch_size=1,
                             shuffle=True, generator=torch.Generator(device='cuda'))
    j=0
    for x,y in tqdm(data_loader):
        x = x.to('cuda')
        n = len(x)
        if y ==1:

         for i in range(n):
            sam = x[i]
            plt.figure()
            plt.axis('off')
            plt.imshow(sam.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
            name = str(f'{j}') + '.png'
            name = os.path.join(path, name)
            plt.savefig(name)
            # plt.savefig(name, dpi=500)
            plt.cla()
            plt.clf()
            j = j + 1

def mnist2png(path="/scratch/private/eunbiyoon/sub_Levy/mnist_1"):
    transform = transforms.Compose([transforms.Resize((28, 28)),
                                      transforms.ToTensor()
                                      ])
    dataset = MNIST('/scratch/private/eunbiyoon/data', train=False, transform=transform, download=True)
    data_loader = DataLoader(dataset, batch_size=1,
                             shuffle=True, generator=torch.Generator(device='cuda'))
    j=0
    for x,y in tqdm(data_loader):
        x = x.to('cuda')
        n = len(x)
        if y==1:
         for i in range(n):
            sam = x[i]
            fig = plt.figure(figsize=(1, 1))
            fig.patch.set_visible(False)
            ax = fig.add_subplot(111)
            ax.set_axis_off()
            ax.imshow(sam.permute(1, 2, 0).cpu(), vmin=0., vmax=1., cmap='gray')
            name = str(f'{j}') + '.png'
            name = os.path.join(path, name)
            plt.savefig(name)
            # plt.savefig(name, dpi=500)
            plt.cla()
            plt.clf()
            j = j + 1

