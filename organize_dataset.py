import os
import sys

from pdb import set_trace as bp

source_path = sys.argv[1]

for file in os.listdir(source_path):

    if os.path.isdir(source_path + file):
        continue

    category = file[0:2]
    target_path = source_path + category

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    source_file = source_path + file
    target_file = target_path + '/' + file

    print('Moving', source_file, 'to', target_file)
    os.rename(source_file, target_file)
