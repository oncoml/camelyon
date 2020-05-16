# map_tumor.py: Identifies tiles that contain tumor regions
# @author: Steve Yang

# System import
import os
import fnmatch
import multiprocessing

# Third-party import
from shapely.geometry import Point
from shapely.geometry import box as Box
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
from shapely.ops import transform
import numpy as np
from PIL import Image, ImageDraw
import cv2

# Local import
import util.tiles as tiles
import util.parse_xml as parse_xml
import util.util as util
import util.integer_mask as integer_mask


def map_tumor(tile_summary, image_dir, xml_path, tile_images_directory):
    """
    Function that will take each tile in a TileSummary object and pass it, along with the the WSI's corresponding XML file to a comparator function.

    Arguments:
       - tile_summary: a TileSummary object derived from WSI's
       - image_dir: location of WSI's
       - xml_path: path to xml files with coordinates of tumorous regions
       - tile_iamges_directory: directory in which the tiles will be saved

    Returns:
       - annotations: list containing positive tiles and x, y coordinates of tumorous regions within the tiles
    """
    annotations = []
    coordinates = parse_xml.parse_xml(xml_path)

    for i in range(len(tile_summary.tiles)):
        tile = tile_summary.tiles[i]

        x, y = compare_tile_segment(coordinates, tile, image_dir, tile_images_directory)

        if x is not None and y is not None:
            temp = [tile.file_title + '_' + str(tile.tile_num), x, y]
            annotations.append(temp)

    return annotations


def compare_tile_segment(coordinates, tile, image_dir, tile_images_directory):
    """
    Compares whether the box bounded by the tile's original coordinates intersects the tumor
    segment indicated in the xml file. Saves all tiles images.
    Arguments:
        coordinates: a list of tumor segments coordinates from xml annotations
        tile: the tile being investigated
        image_dir: directory of original WSI files
        tile_images_directory: directory to which images of tiles will be saved

    Returns:
        Depending on the positivity of the tile:
           - if positive tumor: returns the coordinates on that tile image of the tumor region(s)
           - if negative: returns None, None values
    """
    box = Box(tile.o_c_s, tile.o_r_s, tile.o_c_e, tile.o_r_e)

    for i in range(len(coordinates)):
        # We disregard coordinates with less than 2 xy values
        if len(coordinates[i]) < 3:
            continue

        if not os.path.exists(tile_images_directory + tile.file_title + "_" + str(tile.tile_num) + '.png'):
            save_tile_image(tile, image_dir, tile_images_directory)
            print(tile.file_title + "_", str(tile.tile_num) + ".png image saved")            

        segment = Polygon(coordinates[i])

        if segment.intersects(box):
            segment = transform(reflection(1), segment)
            box = transform(reflection(1), box)

            try:
                overlap = segment.intersection(box)
            except:
                print("Error in ", tile.file_title, " num: ", tile.tile_num, " coord index: ", i)
                continue

            if isinstance(overlap, MultiPolygon):
                x_overlap_shifted_list = []
                y_overlap_shifted_list = []
                for element in overlap:
                    x_overlap_shifted, y_overlap_shifted =  get_xy(tile, element)
                    x_overlap_shifted_list.append(x_overlap_shifted)
                    y_overlap_shifted_list.append(y_overlap_shifted)
                return x_overlap_shifted_list, y_overlap_shifted_list
            else:
                x_overlap_shifted, y_overlap_shifted = get_xy(tile, overlap)

            return x_overlap_shifted, y_overlap_shifted

        else:
            continue

    return None, None
            

def get_xy(tile, polygon):
    x_overlap, y_overlap = polygon.exterior.xy
    x_overlap = [x for x in x_overlap]
    y_overlap = [y for y in y_overlap]
    x_overlap_shifted = [x - tile.o_c_s for x in x_overlap]

    if all(y > 0 for y in y_overlap):
        y_overlap_shifted = [y - tile.o_r_s for y in y_overlap]
    else:
        y_overlap_shifted = [y + tile.o_r_s - 2 for y in y_overlap]
        y_overlap_shifted = [-1 * y for y in y_overlap_shifted]

    return x_overlap_shifted, y_overlap_shifted


def reflection(y0):
    return lambda x, y: (x, 2*y0 - y)


def save_tile_image(tile, image_dir, tile_images_directory):
    wsi_filename = tile.file_title
    tile_pil_image = tiles.tile_to_pil_tile(tile, wsi_filename, image_dir)
    tile_path = tile_images_directory + wsi_filename + '_' + str(tile.tile_num) + '.png'
    tile_pil_image.save(tile_path, 'PNG')
