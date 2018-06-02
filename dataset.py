import cv2
import os
import sys
import numpy as np
import pandas

from pdb import set_trace as bp

import logging

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', \
                    level=logging.INFO, \
                    datefmt='%Y-%m-%d %H:%M:%S')

source_path = sys.argv[1]
if len(sys.argv) > 2:
    test_labels_filename = sys.argv[2]
    test_labels = pandas.read_csv(test_labels_filename)

def build_from_folder_files(source_path):

    for file in os.listdir(source_path):

        if os.path.isdir(source_path + file):
            build_from_folder_files(source_path + file + '/')
        else:
            build_from_file(source_path + file)

def build_from_file(source_file):

    global image_index

    filename = os.path.basename(source_file)
    base, ext = os.path.splitext(filename)

    logging.info('Loading source image ' + source_file)
    x_train[image_index] = cv2.imread(source_file)

    if 'test_labels' not in globals():
        klass = int(base[0:2])
        y_train[image_index, klass] = 1
    else:
        klass = test_labels.loc[test_labels['filename'] == filename]['klass']
        y_train[image_index, int(klass)] = 1

    image_index += 1

train_images_len = 6022
train_images_width = train_images_height = 299
train_classes_len = 83

x_train = np.zeros((train_images_len, train_images_width, train_images_height, 3), dtype=np.uint8)
y_train = np.zeros((train_images_len, train_classes_len), dtype=np.uint8)

image_index = 0

build_from_folder_files(source_path)

logging.info('Tranforming BGR to RGB')
x_train = x_train[...,::-1]

logging.info('Saving X feature matrix')
np.save(source_path + 'x_train', x_train)

logging.info('Saving y target matrix')
np.save(source_path + 'y_train', y_train)

logging.info('Flipping X feature matrix')
x_train_flipped = np.flip(x_train, axis = 2)

logging.info('Saving flipped X feature matrix')
np.save(source_path + 'x_train_flipped', x_train_flipped)
