import os
import sys
import shutil
import logging

from pdb import set_trace as bp

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', \
                    level=logging.INFO, \
                    datefmt='%Y-%m-%d %H:%M:%S')

source_path = sys.argv[1]
target_path = sys.argv[2]
number_of_files = int(sys.argv[3])

def copy_folder_files(source_path, target_path):

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for file in os.listdir(source_path):

        if os.path.isdir(source_path + file):
            copy_folder_files(source_path + file + '/', target_path + file + '/')
        else:
            files = sorted(os.listdir(source_path))[0:number_of_files]

            for selected_file in files:

                source_filename = source_path + selected_file
                target_filename = target_path + selected_file

                logging.info('Copying file ' + source_filename + ' to ' + target_filename)
                shutil.copyfile(source_filename, target_filename)

            return

copy_folder_files(source_path, target_path)
