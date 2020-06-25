# Evaluate slide and generate a patient-level prediction
# @author: Steve Yang

# System imports
import os
import sys

# Module imports
import numpy as np
from openslide import OpenSlide
from scipy import ndimage
import math

sys.path.append('/home/steveyang/projects/camelyon17/')
sys.path.append('/home/steveyang/projects/camelyon17/wsi_into_regions/')

# Local imports
from wsi_into_regions import util
from wsi_into_regions.util.tile_wsi import filter_and_tile
from wsi_into_regions.util.tiles import tile_to_pil_tile 
from inference import inference_file

# Path to for WSI to be evaluated. The format 'patient_xxx_node_yyy.tif'.
# image_dir = '/home/steveyang/Disk/Camelyon17/Train/patient_004/'
# image_file = 'patient_004_node_001.tif'

def evaluate(image_dir, image_file):
    try:
        slide = OpenSlide(image_dir + image_file)
        row = slide.dimensions[1]
        col = slide.dimensions[0]
    except: 
        print("Error opening slide ", image_files)

    tile_summary = filter_and_tile(image_dir, image_file)
    canvas = np.zeros((row, col), dtype=np.byte)

    for tile in tile_summary.tiles:
        pil_tile = tile_to_pil_tile(tile, image_file, image_dir)
        output = inference_file(pil_tile)

        if 1 in output:
            canvas[tile.o_r_s:tile.o_r_e, tile.o_c_s:tile.o_c_e] = output
            
    #canvas = canvas[:, ~np.all(canvas == 0, axis=0)]
    #canvas = canvas[~np.all(canvas == 0, axis=1)]
    nrows = canvas.shape[0]-1
    ncols = canvas.shape[1]-1
    cols_to_del = []
    rows_to_del = []

    for i in range(nrows):
        if np.all(canvas[i, :] == 0):
            if np.all(canvas[i+1, :] == 0):
                rows_to_del.append(i)
    canvas = np.delete(canvas, rows_to_del, axis=0)

    for i in range(ncols):
        if np.all(canvas[:, i] == 0):
            if np.all(canvas[:, i+1] == 0):
                cols_to_del.append(i)
    canvas = np.delete(canvas, cols_to_del, axis=1)

    mat1 = np.ones((3,3), dtype=np.byte)
    labeled_canvas, num_islands = ndimage.label(canvas, structure=mat1)

    dims = []
    for i in range(1, num_islands+1):
        island = np.where(labeled_canvas == i)
        y = island[0][-1] - island[0][0] + 1
        x = island[1][-1] - island[1][0] + 1
        z = max(x, y) 
        dim = z * 0.25
        if dim > 100:
            dims.append(dim)
            print("Island num {} added with dim of {}".format(i, dim))
            if dim >= 256.0:
                print("Big island with with bounding box X: [", 
                        island[1][0], ", ", island[1][-1],
                        "], Y: [", island[0][0], ", ", island[0][-1], "]")
    return dims 
