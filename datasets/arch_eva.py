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
    log_string("-Read datasets by load original txt files, filename: " + str(filelist))
    data_label_filename = os.path.join(ROOT_DIR, 'data', args.folder)
    data_label = np.loadtxt(data_label_filename)
    batch_size = data_label.shape[0]//args.num_points
    seg_labels = data_label[:batch_size*args.num_points, 7]
    data_label = data_label[:batch_size*args.num_points, 0:3]
    xyz_min = np.amin(data_label, axis=0)
    data_label -= xyz_min
        
    data = np.reshape(data_label[...], (batch_size, args.num_points, data_label.shape[-1]))
    seg_labels = np.reshape(seg_labels[...], (batch_size, args.num_points, 1))

    assert(data.shape[1] == label_segs.shape[1])

    foldname = os.path.join(ROOT_DIR, 'data', ,args.folder[:-3]+'_hdf5_%s'%args.num_points)
    file = h5py.File(foldname, 'w')
    file.create_dataset('data', data=data)
    file.create_dataset('label_seg', data=seg_labels)
    file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_points', type=int, default=4096, help='input batch size')
    parser.add_argument('--folder', '-f', help='Path to data folder')
    args = parser.parse_args()
    main(args)