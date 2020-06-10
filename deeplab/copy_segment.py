# Copies positive tiles and segments to form masks

import pickle
from PIL import Image, ImageDraw
from shutil import copyfile

tile_images_dir = '/home/steveyang/projects/camelyon17/tile_images/'
images_dir = '/home/steveyang/projects/camelyon17/tile_images/deeplab/images/'
masks_dir = '/home/steveyang/projects/camelyon17/tile_images/deeplab/masks/'
annot_path = '/home/steveyang/projects/camelyon17/wsi_into_regions/annotations.txt'


def copy_segment(images_dir, masks_dir, annot_path):
    with open(annot_path, 'rb') as f:
        annotations = pickle.load(f)

    annotations = [item for sublist in annotations for item in sublist]

    for i in range(len(annotations)):
        original_path = tile_images_dir + annotations[i][0] + '.png'
        destination_path = images_dir + annotations[i][0] + '.png'
        mask_destination_path = masks_dir + annotations[i][0] + '.png'
        copyfile(original_path, destination_path)

        segment(annotations[i], mask_destination_path)


def segment(tile_annot, destination_path):
    canvas = Image.new('RGB', (1024, 1024), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    if isinstance(tile_annot[1][0], list):
        for i in range(len(tile_annot[1])):
            coordinates = list(zip(tile_annot[1][i], tile_annot[2][i]))
            draw.polygon(coordinates, fill='white')
    else:
        coordinates = list(zip(tile_annot[1], tile_annot[2]))
        draw.polygon(coordinates, fill='white')

    canvas.save(destination_path)

        
