# Discover the mean and std of images

import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from PIL import Image, ImageFile
import glob
import os
import numpy as np

train_dir = '/home/steveyang/projects/camelyon17/tile_images/deeplab/cross_val_0/train/images'
val_dir = '/home/steveyang/projects/camelyon17/tile_images/deeplab/cross_val_0/val/images'


def mean_std(train_dir, val_dir):
    train_dataset = DatasetSegmentation(train_dir)
    val_dataset = DatasetSegmentation(val_dir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)

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

    for images in data_loader:
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


class DatasetSegmentation(data.Dataset):
    def __init__(self, folder_path):
        super(DatasetSegmentation, self).__init__()
        self.img_files = glob.glob(os.path.join(folder_path, '*.png'))

    def __getitem__(self, index):
            img_path = self.img_files[index]
            temp = np.asarray(Image.open(img_path))
            data = temp.copy()
            return torch.from_numpy(data).float()

    def __len__(self):
        return len(self.img_files)
