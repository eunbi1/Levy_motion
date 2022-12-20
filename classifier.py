
# kill error when executing argparse in IPython console

import os

import matplotlib.pyplot as plt
import sampling
import seaborn as sns

import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import argparse
import random
import copy
import torch
import torchvision
import numpy as np
import tqdm
import pandas as pd
import sklearn.metrics as sm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import Unet

from torchsampler import ImbalancedDatasetSampler


def vis(test_accs, confusion_mtxes, labels, figsize=(20, 8)):
    cm = confusion_mtxes[np.argmax(test_accs)]
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%' % p
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'

    fig = plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.plot(test_accs)
    plt.grid(True)

    plt.subplot(1, 2, 2)
    sns.heatmap(cm, annot=annot, fmt='', cmap="Blues")
    plt.savefig('classifier_result.png')
    plt.show()

# class Net(torch.nn.Module):
#     def __init__(self,datasets):
#         super(Net, self).__init__()
#         if datasets =="MNIST":
#             ch=1
#         elif datasets == "CIFAR10":
#             ch =3
#         self.conv1 = torch.nn.Conv2d(ch, 10, kernel_size=5)
#         self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = torch.nn.Dropout2d()
#         self.fc1 = torch.nn.Linear(320, 50)
#         self.fc2 = torch.nn.Linear(50, 10)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, datasets):
        super().__init__()
        if datasets =="MNIST":
            ch=1
        elif datasets == "CIFAR10":
            ch =3
        self.conv1 = nn.Conv2d(ch, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def plot(train_loader):
    counts = torch.zeros((10,))
    for data, target in train_loader:

        counts[target] += 1
    total = counts.sum()
    ratio = counts / total * 100
    ratio = ratio.cpu().numpy()
    ratio = np.round(ratio, 3)
    np.set_printoptions(suppress=True)
    print(ratio)
    plt.bar(range(len(ratio)), ratio, label=f"{ratio}")
    plt.grid()
    plt.legend()
    name = 'MNIST_bar.png'
    plt.savefig(name)


def process(train_loader, test_loader, epochs=1000,
            device='cuda', lr= 0.01, momentum= 0.5
            ,datasets = "MNIST", path=None ):

    model = Net(datasets).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)


    def train(train_loader):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

    def test(test_loader):
        model.eval()
        correct = 0
        targets, preds = [], []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

                correct += pred.eq(target.view_as(pred)).sum().item()

                targets += list(target.cpu().numpy())
                preds += list(pred.cpu().numpy())

            test_acc = 100. * correct / len(test_loader.dataset)
            confusion_mtx = sm.confusion_matrix(targets, preds)
            return test_acc, confusion_mtx

    test_accs, confusion_mtxes = [], []
    if path == None:
        for epoch in tqdm.tqdm(range(1, epochs + 1)):
            train(train_loader)
            test_acc, confusion_mtx = test(test_loader)
            test_accs.append(test_acc)
            confusion_mtxes.append(confusion_mtx)
            print('\rBest test acc = %2.2f%%' % max(test_accs), end='', flush=True)
        name = datasets+ 'classifier.pth'
        torch.save(model.state_dict(), name)
        vis(test_accs, confusion_mtxes, classe_labels)
    else:
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        model.eval()
        test_acc, confusion_mtx = test(test_loader)
        test_accs.append(test_acc)
        confusion_mtxes.append(confusion_mtx)
        vis(test_accs, confusion_mtxes, classe_labels )
#
# batch_size =200
# image_size =32
# transform = Compose([
#     Resize(image_size),
#     ToTensor()])
# num_classes = 10
# classe_labels = range(num_classes)
# transformer = transforms.Compose([transforms.Resize((32, 32)),
#                                   transforms.ToTensor()])
#
# train_dataset = torchvision.datasets.CIFAR10('.', train=True, download=True, transform=transform)
# train_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10('.', train=True, transform=transformer),
#     batch_size=batch_size, shuffle=False,generator=torch.Generator(device='cuda'))
# test_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10('.', train=False, transform=transformer),
#     batch_size=batch_size, shuffle=False,generator=torch.Generator(device='cuda'))
#
# process(train_loader, test_loader, datasets="CIFAR10", epochs =10000)

def ratio_test(path="/scratch/private/eunbiyoon/sub_Levy/2MNISTbatch128lr0.0001ch128ch_mult[1, 2, 2, 2]num_res2dropout0.1clamp201.8_0.1_20/ckpt/MNISTbatch128ch128ch_mult[1, 2, 2, 2]num_res2dropout0.1clamp20_epoch1248_1.8_0.1_20.pth",
               alpha=1.8, datasets='MNIST', device='cuda',max_iter =100,conditional=True, num_classes=10,
               ch=128, ch_mult=[1,2,2,2], num_res_blocks=2, resolution=28, name=''):

    if conditional== False:
        num_classes = None
    model = Net(datasets).to(device)

    if datasets == "CIFAR10":
        ckpt = torch.load("/scratch/private/eunbiyoon/sub_Levy/cifar10_classifier.pth",
                      map_location=device)
        num_classes = 10
    elif datasets == "MNIST":
        ckpt = torch.load("/scratch/private/eunbiyoon/sub_Levy/MNISTclassifier.pth", map_location=device)
        num_classes = 10
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    #torch.manual_seed(0)

    c = np.zeros((num_classes,))

    with torch.no_grad():
        for epoch in tqdm.tqdm(range(max_iter)):
            samples = sampling.sample(alpha=alpha, path=path, conditional=conditional,num_classes=num_classes,
                                      sampler='ode_sampler', batch_size=1000, num_steps=20, LM_steps=50, name='',
                                      dir_path='/scratch/private/eunbiyoon/sub_Levy/sample',
                                      datasets=datasets, ch=ch, ch_mult=ch_mult, num_res_blocks=num_res_blocks,
                                      resolution=resolution,y=1)
            samples = samples.to(device)

            output = model(samples)

            pred = output.max(1,keepdim=True)[1]

            output, counts = torch.unique(pred, sorted=True, return_counts=True)

            for i, count in zip(output,counts):
                c[i]+= count
    total = c.sum()
    ratio = c/total*100
    ratio = np.round(ratio, 3)
    np.set_printoptions(suppress=True)
    print(ratio)
    plt.bar(range(len(ratio)), ratio, label=f"{alpha}{ratio}")
    plt.grid()
    plt.legend()
    name2 = name+f'0.1imbalanced{alpha}{datasets}bar.png'
    plt.savefig(name2)

# ratio_test(path="/scratch/private/eunbiyoon/sub_Levy/[0, 0.1, 0, 0, 0, 0, 0, 0, 0.9, 0]conditionalMNISTbatch128lr0.0001ch128ch_mult[1, 2, 2, 2]num_res4dropout0.1clamp202.0_0.1_20/ckpt/MNISTbatch128ch128ch_mult[1, 2, 2, 2]num_res4dropout0.1clamp20_epoch1405_2.0_0.1_20.pth",
#                alpha=2.0, datasets='MNIST', device='cuda',
#                ch=128, ch_mult=[1,2,2,2], num_res_blocks=2, resolution=28)
# ratio_test(path="/scratch/private/eunbiyoon/sub_Levy/[0, 0.1, 0, 0, 0, 0, 0, 0, 0.9, 0]conditionalMNISTbatch128lr0.0001ch128ch_mult[1, 2, 2, 2]num_res4dropout0.1clamp201.8_0.1_20/ckpt/MNISTbatch128ch128ch_mult[1, 2, 2, 2]num_res4dropout0.1clamp20_epoch1405_1.8_0.1_20.pth",
#                alpha=1.8, datasets='MNIST', device='cuda',
#                ch=128, ch_mult=[1,2,2,2], num_res_blocks=2, resolution=28)

# ratio_test(path="/scratch/private/eunbiyoon/sub_Levy/[0, 0.05, 0, 0, 0, 0, 0, 0, 0.95, 0]MNISTbatch64lr0.0001ch128ch_mult[1, 2, 2, 2]num_res4dropout0.1clamp201.5_0.1_20/ckpt/MNISTbatch64ch128ch_mult[1, 2, 2, 2]num_res4dropout0.1clamp20_epoch595_1.5_0.1_20.pth",
#                alpha=1.5, datasets='MNIST', device='cuda',
#                ch=128, ch_mult=[1,2,2,2], num_res_blocks=2, resolution=28)

