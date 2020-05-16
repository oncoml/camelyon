# Copies tiles without tumor regions to the appropriate directories

import shutil
from shutil import copyfile
import os
from random import choice

tiles_directory = '/home/steveyang/projects/camelyon17/tile_images/'
comparator_train_directory = '/home/steveyang/projects/camelyon17/tile_images/resnet/cross_val_0/train/tumor/'
comparator_val_directory = '/home/steveyang/projects/camelyon17/tile_images/resnet/cross_val_0/val/tumor/'
target_train_directory = '/home/steveyang/projects/camelyon17/tile_images/resnet/cross_val_0/train/nontumor/'
target_val_directory = '/home/steveyang/projects/camelyon17/tile_images/resnet/cross_val_0/val/nontumor/'

def copy_negative_tiles(tiles_directory, comparator_train_directory, comparator_val_directory, target_train_directory, target_val_directory):
    
    files_all = os.listdir(tiles_directory)
    files_negative = []
    files_val = []
    files_train = []

    for file in files_all:
        if file not in os.listdir(comparator_train_directory) and file not in os.listdir(comparator_val_directory) and file != 'resnet':
            files_negative.append(file)

    num_files = len(files_negative)
    num_train_files = round(num_files * 0.85)
    num_val_files = round(num_files * 0.15)

    while len(files_train) < num_train_files:
        name = choice(files_negative)
        if name not in files_train:
            files_train.append(name)
    
    while len(files_val) < num_val_files:
        name = choice(files_negative)
        if name not in files_val and name not in files_train:
            files_val.append(name)

    copy_file(files_train, tiles_directory, target_train_directory)
    copy_file(files_val, tiles_directory, target_val_directory)


def copy_file(files_list, tiles_directory, target_directory):
    for file in files_list:
        copyfile(tiles_directory + file, target_directory + file)
