import os
from random import choice
import pickle
from shutil import copyfile 
from copy_segment import segment


annotations_path = '/home/steveyang/projects/camelyon17/wsi_into_regions/annotations.txt'
tile_images_dir = '/home/steveyang/projects/camelyon17/tile_images/'
target_root_dir = '/home/steveyang/projects/camelyon17/tile_images/deeplab/'


def train_val_test_split(annotations_path, tile_images_dir, target_root_dir, test_split=0.00, val_split=0.15, num_cross_val=1):
    """
    A function that divides a list of images into training, validation and test sets. For use with Pytorch training

    Arguments:
       - annotations: a list containing tiles and tumor coordinates, in binary
       - tile_images_dir: directory containing all the tile images
       - target_root_dir: directory to which train, val subdirectories are located
       - test_split: portion of images reserved for test set
       - val_split: portion of images reserved for validation set, from original number of images
       - num_cross_val: fold cross-validation

    Returns:
       - None
    """
    train_split = 1 - val_split - test_split

    with open(annotations_path, 'rb') as f:
        annotations = pickle.load(f)

    annotations = [item for sublist in annotations for item in sublist]
    test_files = []

    # Separate out test files from the rest
    files = [a[0] for a in annotations if a is not None]

    num_all_files = len(files)
    num_test_files = round(num_all_files * test_split)
    num_val_files = round(num_all_files * val_split)

    while len(test_files) < num_test_files:
        name = choice(files)
        if name not in test_files:
            test_files.append(name)

    # Process split for training and validation, repeat for number of cross-validation
    if not os.path.exists(target_root_dir):
        os.makedirs(target_root_dir)

    for i in range(num_cross_val):
        train_val_directory = os.path.join(target_root_dir, 'cross_val' + '_' + str(i))
        if not os.path.exists(train_val_directory):
            os.makedirs(train_val_directory)

        train_directory = os.path.join(train_val_directory, 'train' + '/')
        train_images_directory = os.path.join(train_directory, 'images' + '/')
        train_masks_directory = os.path.join(train_directory, 'masks' + '/')
        if not os.path.exists(train_directory):
            os.makedirs(train_directory)
            os.makedirs(train_images_directory)
            os.makedirs(train_masks_directory)

        val_directory = os.path.join(train_val_directory, 'val' + '/')
        val_images_directory = os.path.join(val_directory, 'images' + '/')
        val_masks_directory = os.path.join(val_directory, 'masks' + '/')
        if not os.path.exists(val_directory):
            os.makedirs(val_directory)
            os.makedirs(val_images_directory)
            os.makedirs(val_masks_directory)

        test_directory = os.path.join(train_val_directory, 'test' + '/')
        test_images_directory = os.path.join(test_directory, 'images' + '/')
        test_masks_directory = os.path.join(test_directory, 'masks' + '/')
        if test_split > 0.0:
            if not os.path.exists(test_directory):
                os.makedirs(test_directory)
                os.makedirs(test_images_directory)
                os.makedirs(test_masks_directory)

        val_files = []
        while len(val_files) < num_val_files:
            name = choice(files)
            if name not in test_files and name not in val_files:
                val_files.append(name)

        train_files = [item for item in files if (item not in val_files and item not in test_files)]

        copy_images(train_files, tile_images_dir, train_images_directory)
        copy_images(val_files, tile_images_dir, val_images_directory)
        if test_split > 0.0:
            copy_images(test_files, tile_images_dir, test_images_directory)

        for entry in train_files:
            for i in range(len(annotations)):
                if entry == annotations[i][0]:
                    mask_destination_path = train_masks_directory + annotations[i][0] + '.png'
                    segment(annotations[i], mask_destination_path)

        for entry in val_files:
            for i in range(len(annotations)):
                if entry == annotations[i][0]:
                    mask_destination_path = val_masks_directory + annotations[i][0] + '.png'
                    segment(annotations[i], mask_destination_path)
        
        if test_split > 0.0:
            for entry in test_files:
                for i in range(len(annotations)):
                    if entry == annotations[i][0]:
                        mask_destination_path = test_masks_directory + annotations[i][0] + '.png'
                        segment(annotations[i], mask_destination_path)


def copy_images(files_list, copy_from_folder, copy_to_folder):
    """
    Copies tile images from copy_from_folder to copy_to_folder according to the list listed
    in annotation_file_path. Used for dataloader in Pytorch.
    """
    for file in files_list:
        try:
            copyfile(copy_from_folder + file + '.png', copy_to_folder + file + '.png')
        except FileNotFoundError:
            print("Can not find ", copy_from_folder + file + '.png')
            break
