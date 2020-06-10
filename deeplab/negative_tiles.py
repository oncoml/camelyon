# Copies tiles with and without tumor regions to the appropriate directories

import shutil
from shutil import copyfile
import os
from random import choice
from PIL import Image, ImageDraw 

tiles_directory = '/home/steveyang/projects/camelyon17/tile_images/'
train_directory = '/home/steveyang/projects/camelyon17/tile_images/deeplab/cross_val_neg/train/'
val_directory = '/home/steveyang/projects/camelyon17/tile_images/deeplab/cross_val_neg/val/'
target_train_directory = os.path.join(train_directory, 'images/') 
target_val_directory = os.path.join(val_directory, 'images/') 

def copy_negative_tiles(tiles_directory, train_directory, val_directory, target_train_directory, target_val_directory):
    
    files_all = os.listdir(tiles_directory)
    comparator_train_directory = os.path.join(tiles_directory, 'deeplab/cross_val_0/train/images/')
    comparator_val_directory = os.path.join(tiles_directory, 'deeplab/cross_val_0/val/images/')
    files_negative = []
    files_val = []
    files_train = []

    for file in files_all:
        if file not in os.listdir(comparator_train_directory) and file not in os.listdir(comparator_val_directory) and file != 'deeplab' and file!= 'resnet':
            files_negative.append(file)

    num_files = 1500
    num_train_files = round(num_files * 0.85)
    num_val_files = round(num_files * 0.15)

    while len(files_train) < num_train_files:
        name = choice(files_negative)
        if name not in files_train:
            files_train.append(name)
            canvas = Image.new('RGB', (1024, 1024), (0, 0, 0))
            draw = ImageDraw.Draw(canvas)
            canvas.save(train_directory + 'masks/' + name, 'png') 
    
    while len(files_val) < num_val_files:
        name = choice(files_negative)
        if name not in files_val and name not in files_train:
            files_val.append(name)
            canvas = Image.new('RGB', (1024, 1024), (0, 0, 0))
            draw = ImageDraw.Draw(canvas)
            canvas.save(val_directory + 'masks/' + name, 'png')

    copy_file(files_train, tiles_directory, target_train_directory)
    copy_file(files_val, tiles_directory, target_val_directory)

def copy_file(files_list, tiles_directory, target_directory):
    for file in files_list:
        copyfile(tiles_directory + file, target_directory + file)
