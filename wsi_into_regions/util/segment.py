### Segmentation of nuclei and stroma from tiles. For nuclei segmentation we use
### a deep-learning model from Kemeng Chen
### @author: Steve Yang

# Standard import
import os
import multiprocessing

# Third-party import
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageChops, ImageOps, ImageFilter

# Local import
from cell_nuclei_detection_segmentation.util import nuclei_ds_util as nuclei_util
from cell_nuclei_detection_segmentation.util import run_restored_model as run_restored_model
import util.tiles as tiles
import util.filter as filter
import util.util as util


def segment_nuclei(tile, wsi_directory, nuclei_detect_directory, tile_images_directory, model_directory, model_name):
    '''
    Uses multiprocessing to segment nuclei and stroma into masks images
    
    Arguments:
       - tile_summary_list: a list consisting of TileSummary objects, one per WSI
       - nuclei_detect_directory: directory that will contain images of detected nuclei
       - nuclei_detect_filetype: filetype of images of detected nuclei
       - wsi_directory: directory containing the WSI images
       - model_name: model name for nuclei detection
       - model_directory: directory containing model
       - masked_nuclei_directory: directory that will contain images of masked nuclei
       - masked_nuclei_filetype: filetype of images of masked nuclei
       - masked_stroma_directory: directory that will contain images of masked stroma
       - masked_stroma_filetype: filetype of images of masked stroma
       - tile_images_directory: directory that will contain images of tiles
       - tile_images_filetype: filetype of images of tiles
    
    Returns:
       - None, but saves all masks in the appropriate folders    
    '''
    model_path = model_directory + model_name
    model = run_restored_model.restored_model(model_path, model_directory)
    patch_size = 128
    stride = 16

    wsi_filename = tile.file_title
    tile_pil_img = tiles.tile_to_pil_tile(tile, wsi_filename, wsi_directory)
    tile_path = tile_images_directory + wsi_filename + '_' + str(tile.tile_num) + '.png'
    tile_pil_img.save(tile_path, 'PNG')

    cv_tile = tiles.tile_to_np_tile(tile, wsi_filename, wsi_directory)
    cv_tile = cv_tile[:, :, ::-1]

    batch_group, shape = nuclei_util.preprocess(cv_tile, patch_size, stride)
    mask_list = nuclei_util.sess_interference(model, batch_group)
    c_mask = nuclei_util.patch2image(mask_list, patch_size, stride, shape)
    c_mask = cv2.medianBlur((255 * c_mask).astype(np.uint8), 3)
    c_mask = c_mask.astype(np.float) / 255
    thr = 0.6
    c_mask[c_mask < thr] = 0
    c_mask[c_mask >= thr] = 1
    _, gray_map = nuclei_util.center_edge(c_mask, cv_tile)
    cv2.imwrite(os.path.join(nuclei_detect_directory, wsi_filename + '_' + str(tile.tile_num) + '.png'), gray_map)
