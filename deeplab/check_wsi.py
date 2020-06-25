# Checks that all WSI beloging to a patient is ok

import os
from openslide import OpenSlide

root_dir = '/home/steveyang/Data2T/camelyon17/'

def check_wsi(num):
    folder_name = 'patient_' + num
    folder_dir = root_dir + folder_name + '/'
    files_list = os.listdir(folder_dir)

    for filename in files_list:
        file_path = folder_dir + filename
        try:
            slide = OpenSlide(file_path)
            print("Slide is ok: {}".format(filename))
        except:
            print("Error opening {}".format(filename))
            continue
