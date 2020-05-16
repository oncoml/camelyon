# Tile WSI (Whole Slide Imaging) files and maps tumor regions based on XML coordinates.  
# Name format of WSI slides are consistent with those found in Camelyon17 Challenge.
# @author: Steve Yang

# System imports
import os
import fnmatch
import pickle

# Local imports
from util import tile_wsi, map_tumor

# Directories for WSI data and saving tiles. WSI_DIRECTORY should contain WSI files of 
# the format 'patient_xxx_node_yyy.tif' as per Camelyon17 data.
IMAGES_DIR = '/home/steveyang/Disk/Camelyon17/Train/'
TILE_IMAGES_DIRECTORY = '/home/steveyang/projects/camelyon17/tile_images_2/'

# Path to positive slides text file that lists WSI slides that are positive for tumor
POSITIVE_SLIDES_PATH = '/home/steveyang/projects/camelyon17/positive_slides.txt'

# Path to saved annotations file
ANNOT_DIR = IMAGES_DIR

# Parameters for tumor annotations and coloring
XML_DIRECTORY = '/home/steveyang/Disk/Camelyon17/Train/lesion_annotations/'


def main():
    annotations = []

    with open(POSITIVE_SLIDES_PATH, 'r') as f:
        positive_slides = f.read().splitlines()

    for slide in positive_slides:
        if slide == 'patient_017_node_4.tif':
            continue
        try:
            image_dir = os.path.join(IMAGES_DIR, slide[:-11] + '/')
            image_path = os.path.join(image_dir, slide)
            xml_path = os.path.join(XML_DIRECTORY, slide[:-3] + 'xml')
            tile_summary = tile_wsi.filter_and_tile(image_dir, slide)
            annotations.append(map_tumor.map_tumor(tile_summary, image_dir, xml_path, TILE_IMAGES_DIRECTORY))

        except FileNotFoundError:
            continue

    with open('annotations.txt', 'wb') as file:
        pickle.dump(annotations, file)

if __name__ == '__main__':
    main()
