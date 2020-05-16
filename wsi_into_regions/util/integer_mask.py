### Converting RGB images into 2d labels or 1-hot encodings. Derived from Foivos Diakogiannis
### and Delphi Fan's work.

# System import
import os
import multiprocessing
import fnmatch

# Third-party import
import numpy as np
import cv2

# Local import


def mask_to_integer(mask_directory, mask_onehot_directory, color_scheme):
    '''
    Multiprocessed function to map colored masks into single-numbered classes. For example, for tumored nuclei regions that is colored (255, 0, 0) or red, convert all red pixels to the integer "1". This is done by converting (255, 0, 0) to its corresponding gray value (29) and putting 1 in (255,0,0) pixels though a lookup table (lut)
    
    Arguments:
       - mask_directory: directory containing images of colored masks
       - mask_onehot_directory: directory that will contain converted colored masks'
       - color_scheme: color scheme defined in main.py
       
    Returns:
       - None, but saves converted images in mask_onehot_directory folder
    '''
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_processes)
    mask_file_list = []

    for filename in os.listdir(mask_directory):
        if fnmatch.fnmatch(filename, '._*') == False and fnmatch.fnmatch(filename, '*.png') == \
                True and os.path.isdir(filename) == False:
            mask_file_list.append(filename)

    for mask in mask_file_list:
        pool.starmap(rgb_to_1h_label, [(mask, mask_directory, mask_onehot_directory, color_scheme)])


def convert_rgb_gray(rgb):
    '''
    Converts RGB value to its corresponding grayscale version
    
    Arguments:
       - bgr: a tuple containing BGR values
       
    Returns:
       - a grayscale value
    '''
    assert len(rgb) == 3
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]
    return round(0.299*r + 0.587*g + 0.114*b)


def rgb_to_1h_label(mask_name, mask_directory, mask_onehot_directory, color_scheme):
    '''
    Converts masked BGR images to their corresponding integer version
    
    Arguments:
       - mask_name: name of mask being processed
       - mask_directory: directory containing masks to be processed
       - mask_onehot_directory: directory that will contain the converted images
       - color_scheme: color scheme defined in main.py
    
    Returns:
       - None, saves converted images in mask_onehot_directory
    '''
    mask_path = mask_directory + mask_name
    np_mask = cv2.imread(mask_path)
    grays = []
    lut = np.ones(256, dtype = np.uint8) * 255

    for item in color_scheme:
        gray = convert_rgb_gray(item)
        grays.append(gray)

    lut[grays] = np.arange(len(color_scheme), dtype = np.uint8)

    image_out = cv2.LUT(cv2.cvtColor(np_mask, cv2.COLOR_BGR2GRAY), lut)

    image_out_path = mask_onehot_directory + mask_name
    cv2.imwrite(image_out_path, image_out)
