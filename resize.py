import cv2
import os
import sys

from pdb import set_trace as bp

import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', \
                    level=logging.INFO, \
                    datefmt='%Y-%m-%d %H:%M:%S')

source_path = sys.argv[1]
target_path = sys.argv[2]

def resize_folder_files(source_path, target_path, target_width, target_height):

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for file in os.listdir(source_path):

        if os.path.isdir(source_path + file):
            resize_folder_files(source_path + file + '/', target_path + file + '/', target_width, target_height)
        else:
            source_filename = source_path + file
            target_filename = target_path + file

            resize_file(source_filename, target_filename, target_width, target_height)

def resize_file(source_file, target_file, target_width, target_height):

    logging.info('Loading source image ' + source_file)
    source_image = cv2.imread(source_file)

    logging.info('Resizing source image ' + source_file)
    target_image = cv2.resize(source_image, (target_width, target_height), interpolation = cv2.INTER_AREA)

    logging.info('Saving target image ' + target_file)
    cv2.imwrite(target_file, target_image)

target_width = target_height = 299
resize_folder_files(source_path, target_path, target_width, target_height)
