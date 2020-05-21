# Discover the mean and std of images

import torch
from train import load_data 

train_dir = '/home/steveyang/projects/camelyon17/tile_images/resnet/cross_val_0/train'
val_dir = '/home/steveyang/projects/camelyon17/tile_images/resnet/cross_val_0/val'


def mean_std(train_dir, val_dir):
    train_loader, val_loader = load_data(train_dir, val_dir)

    mean_train, std_train = process(train_loader)
    mean_val, std_val = process(val_loader)

    print('mean_train: ', mean_train)
    print('std_train: ', std_train)
    print('mean_val: ', mean_val)
    print('std_val: ', std_val)


def process(data_loader):

    R_means = []
    G_means = []
    B_means = []

    R_stds = []
    G_stds = []
    B_stds = []

    for images, labels in data_loader:
        for i in range(len(images)):
            R_means.append(torch.mean(images[i][0]))
            G_means.append(torch.mean(images[i][1]))
            B_means.append(torch.mean(images[i][2]))
            R_stds.append(torch.std(images[i][0]))
            G_stds.append(torch.std(images[i][1]))
            B_stds.append(torch.std(images[i][2]))

    mean_R = torch.mean(torch.tensor(R_means)).item()
    mean_G = torch.mean(torch.tensor(G_means)).item()
    mean_B = torch.mean(torch.tensor(B_means)).item()
    
    std_R = torch.mean(torch.tensor(R_stds)).item()
    std_G = torch.mean(torch.tensor(G_stds)).item()
    std_B = torch.mean(torch.tensor(B_stds)).item()

    return (mean_R, mean_G, mean_B), (std_R, std_G, std_B)

