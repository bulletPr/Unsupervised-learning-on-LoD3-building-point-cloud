#
#
#      0=================================0
#      |    Project Name                 |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Implements read ArchDatase
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO - 2021/1/13 21:03
#
#
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import h5py
import argparse
import numpy as np

from datetime import datetime
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import tensorflow as tf

def load_h5_seg(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][...].astype(np.float32)
    label = f['label_seg'][...].astype(np.int64)
    return data, label

def main(args):
    label2color = {
        0: [255, 255, 255],  # white
        1: [0, 0, 255],  # blue
        2: [128, 0, 0],  # maroon
        3: [255, 0, 255],  # fuchisia
        4: [0, 128, 0],  # green
        5: [255, 0, 0],  # red
        6: [128, 0, 128],  # purple
        7: [0, 0, 128],  # navy
        8: [128, 128, 0],  # olive
        9: [128, 128, 128]
    }
    data_label_filename = os.path.join(ROOT_DIR, 'data', args.folder)
    print("-Read datasets by load original txt files, filename: " + str(data_label_filename))
    data, label = load_h5_seg(data_label_filename)
    print("Original points shape: " + str(data.shape) + "label shape: " + str(label.shape))
    points = data[:,:,0:3]
    points = np.array(points).reshape((-1, 3))
    label = np.array(label).astype(int).flatten()
    print("Reshaped points shape: " + str(points.shape) + "label shape: " + str(label.shape))
    assert(points.shape[0] == label.shape[0])
    save_dir = os.path.join(ROOT_DIR, 'data', args.outpath)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fout = open(os.path.join(save_dir, 'Scene_A_pred.obj'),'w')
    for i in range(label.shape[0]):
        color = label2color[label[i]]
        fout.write('v %f %f %f %d %d %d\n' % (
                    points[i, 0], points[i, 1], points[i, 2], color[0], color[1],
                    color[2]))
    print("Exported sparse pcd to " + str(os.path.join(ROOT_DIR, 'data', args.outpath, 'Scene_A_pred.obj')))
    fout.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', help='Path to data folder')
    parser.add_argument('--outpath', '-o', help='Path to data folder')
    args = parser.parse_args()
    main(args)