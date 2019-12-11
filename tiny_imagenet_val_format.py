# script modified from   https://raw.githubusercontent.com/tjmoon0104/pytorch-tiny-imagenet/master/val_format.py

import io
import glob
import os
from shutil import move
from os.path import join
from os import listdir, rmdir

target_folder = './datasets/tinyimagenet/val/'

val_dict = {}
with open('./datasets/tinyimagenet/val/val_annotations.txt', 'r') as f:
    for line in f.readlines():
        split_line = line.split('\t')
        val_dict[split_line[0]] = split_line[1]

paths = glob.glob('./datasets/tinyimagenet/val/images/*')
for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    if not os.path.exists(target_folder + str(folder)):
        os.mkdir(target_folder + str(folder))
        os.mkdir(target_folder + str(folder) + '/images')

for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    dest = target_folder + str(folder) + '/images/'+ str(file)
    move(path, dest)

rmdir('./datasets/tinyimagenet/val/images')
