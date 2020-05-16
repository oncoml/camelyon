# Saves tile images overlaid with tumor regions (for positive tiles only)
# @author: Steve Yang

# System imports
import re
import pickle

# Third-party import
from PIL import Image, ImageDraw

# Local imports

# Directories for tile images and saving tiles
tile_images_dir = '/home/steveyang/projects/camelyon17/tile_images/'
target_dir = '/home/steveyang/Data2T/tile_images_masked_2/'

def check_tile_dir(annotations_file, tile_images_dir, target_dir):
    with open(annotations_file, 'rb') as f:
        annotations = pickle.load(f)

    annotations =  [item for sublist in annotations for item in sublist]
    for entry in annotations:
        check_tile(entry, tile_images_dir, target_dir)


def check_tile(entry, tile_images_dir, target_dir):
    tile_file = entry[0]
    pil_image  = Image.open(tile_images_dir + tile_file + '.png')
    pil_image_2 = pil_image.copy()
    draw = ImageDraw.Draw(pil_image)
    
    if isinstance(entry[1][0], list):
        for i in range(len(entry[1])):
            z_shifted = list(zip(entry[1][i], entry[2][i]))
            draw.polygon(z_shifted, fill = "wheat")
            image_blend = Image.blend(pil_image, pil_image_2, 0.5)
    else:
        z_shifted = list(zip(entry[1], entry[2]))
        draw.polygon(z_shifted, fill = "wheat")
        image_blend = Image.blend(pil_image, pil_image_2, 0.5)
    
    image_blend.save(target_dir + tile_file + '.png')
    print(tile_file, " masked image saved")
