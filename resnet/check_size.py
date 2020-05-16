# Checks tile images to ensure all are of the same sizes
# To be run before split.py

import os
from PIL import Image

tile_images_dir = '/home/steveyang/projects/camelyon17/tile_images/'

def check_sizes(tile_images_dir):
    tile_images_files = [f for f in os.listdir(tile_images_dir) if os.path.isfile(tile_images_dir + f)]

    for tile in tile_images_files:
        np_tile = np.asarray(Image.open(tile_images_dir + tile))
        try:
            if np_tile.shape[0] != np_tile.shape[1]:
                print("Tile shape error in ", tile)
        except:
            print("Tile file error in ", tile)
            continue
