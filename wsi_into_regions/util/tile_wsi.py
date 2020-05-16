### tile_wsi.py: Tile WSI slides by filtering and identifying tiles containing cells
### @author: Steve Yang

# Standard import
import os

# Third-party modules import
import numpy as np

# Local import
import util.slide as slide
import util.util as util
import util.filter as filter
import util.tiles as tiles


def filter_and_tile(wsi_directory, wsi):
    '''
    Filtering and tiling a WSI slide
    Arguments:
       - wsi_directory: directory where the WSI images are stored
       - wsi: a WSI file

    Returns:
       - tile_summary: a TileSummary object of the WSI
    '''
    scaled_wsi, large_w, large_h, small_w, small_h = slide.slide_to_scaled_pil_image(wsi_directory + wsi)
    np_wsi = util.pil_to_np_rgb(scaled_wsi)
    np_wsi_masked = general_filter(np_wsi)

    tile_summary = tiles.summary_and_tiles(wsi, np_wsi_masked, large_w, large_h, small_w, small_h,display=False, save_summary=False, save_data=False, save_top_tiles=False)

    return tile_summary


def general_filter(np_image):
    '''
    A simple function to filter a Numpy image of a tile
    Arguments:
        - np_image: a Numpy image of dimension [col, row, 3]

    Returns:
        - np_image_masked: a filtered Numpy image of same dimension as input
    '''
    mask_not_gray = filter.filter_grays(np_image)
    mask_not_green = filter.filter_green_channel(np_image, green_thresh=200, avoid_overmask=True,
                                                 overmask_thresh=90, output_type="bool")
    mask_gray_green = mask_not_gray & mask_not_green
    np_image_masked = util.mask_rgb(np_image, mask_gray_green)
    return np_image_masked
