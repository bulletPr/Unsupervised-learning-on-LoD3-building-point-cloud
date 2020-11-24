#
#
#      0=================================0
#      |    Project Name                 |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Implements
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO - 11/24/2020 16:45 PM
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
sys.path.append(BASE_DIR)

sys.path.append(os.path.join(BASE_DIR, 'tf_ops/sampling'))
from tf_sampling import farthest_point_sample, gather_point

import tensorflow as tf

def load_h5_seg(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][...].astype(np.float32)
    label = f['label_seg'][...].astype(np.int64)
    return data, label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', '-f', help='Path to data folder')
    
    root = args.folder if args.folder else os.path.join(ROOT_DIR, 'data', 'arch_hdf5_8196')

    folders = [os.path.join(root, folder) for folder in ['test', 'train']]

    for folder in folders:
        datasets = os.listdir(folder)
        for dataset_idx, dataset in enumerate(datasets):
            
            filename_txt = os.path.join(folder, dataset)
            print('{}-Loading {}...'.format(datetime.now(), filename_txt))

            data, label_seg = load_h5_seg(filename_h5)
            assert(data.shape[1] == label_seg.shape[1])

            points_sampled = gather_point(xyz, farthest_point_sample(4096, data)) # (batch_size, npoint, 3)
            print(points_sampled.shape)
            
            labels_sampled = tf.gather_nd(labels_seg, indices=indices, name='labels_sampled')
            print(labels_sampled.shape)

            sampled_filename = os.path.join(folder, dataset + '_4096_sampled.h5')
            file = h5py.File(sampled_filename, 'w')
            file.create_dataset('data', data=points_sampled)
            file.create_dataset('label_seg', data=labels_sampled)
            file.close()


if __name__ == '__main__':
    main()