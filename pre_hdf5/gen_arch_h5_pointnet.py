
#
#
#      0=================================0
#      |    Project Name                 |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Implements: BATCH TO HDF5 FORTMAT
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO - 2020/10/21 5:56 PM 
#
#
import os
import numpy as np
import sys
import random
import argparse
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_utils

import common

parse = argparse.ArgumentParser()
parse.add_argument('--split_file', default='')
parse.add_argument('--block_size', default=1.0)
parse.add_argument('--stride', type=float, default=0.5)
parse.add_argument('--split', default='train')
parse.add_argument('--data_dir', default='arch')
args = parse.parse_args()

# Constants
NUM_POINT = 4096
H5_BATCH_SIZE = 1000
data_dim = [NUM_POINT, 6]
label_dim = [NUM_POINT]
data_dtype = np.float32
label_dtype = np.int8

if args.data_dir == 'arch':
    output_dir = os.path.join(ROOT_DIR, 'data', 'arch_{}m_pointnet_hdf5_data'.format(args.block_size, args.stride), args.split)
else:
    output_dir = os.path.join(ROOT_DIR, 'data', '{}_{}m_pointnet_hdf5_data'.format(args.data_dir, args.block_size, args.stride), args.split)

if not os.path.exists(output_dir):
    os.mkdir(os.path.join(ROOT_DIR, 'data', '{}_{}m_pointnet_hdf5_data'.format(args.data_dir, args.block_size, args.stride)))
    os.mkdir(output_dir)
output_filename_prefix = os.path.join(output_dir, 'h5_data_all')

# --------------------------------------
#           BATCH WRITE TO HDF5
# --------------------------------------

batch_data_dim = [H5_BATCH_SIZE] + data_dim #[1000, 4096, 6]
batch_label_dim = [H5_BATCH_SIZE] + label_dim #[1000, 4096]
h5_batch_data = np.zeros(batch_data_dim, dtype = np.float32) #(1000, 4096, 6)
h5_batch_label = np.zeros(batch_label_dim, dtype = np.uint8) #(1000, 4096)
buffer_size = 0  # state: record how many samples are currently in buffer
h5_index = 0 # state: the next h5 file to save

def insert_batch(data, label, last_batch=False):
    global h5_batch_data, h5_batch_label
    global buffer_size, h5_index
    data_size = data.shape[0]
    # If there is enough space, just insert
    if buffer_size + data_size <= h5_batch_data.shape[0]:
        h5_batch_data[buffer_size:buffer_size+data_size, ...] = data
        h5_batch_label[buffer_size:buffer_size+data_size] = label
        buffer_size += data_size
        log_string("now in enough space location, store data in memory")
    else: # not enough space
        capacity = h5_batch_data.shape[0] - buffer_size
        assert(capacity>=0)
        if capacity > 0:
           h5_batch_data[buffer_size:buffer_size+capacity, ...] = data[0:capacity, ...]
           h5_batch_label[buffer_size:buffer_size+capacity, ...] = label[0:capacity, ...]
        # Save batch data and label to h5 file, reset buffer_size
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        pc_utils.save_h5(h5_filename, h5_batch_data, h5_batch_label, data_dtype, label_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, h5_batch_data.shape[0]))
        h5_index += 1
        buffer_size = 0
        # recursive call
        log_string("now insert rest data to h5 batch again" + str(data[capacity:, ...].shape))
        insert_batch(data[capacity:, ...], label[capacity:, ...], last_batch)
    if last_batch and buffer_size > 0:
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        pc_utils.save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...], data_dtype, label_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, buffer_size))
        h5_index += 1
        buffer_size = 0
        log_string("in last batch!")
    return


LOG_FOUT = open(os.path.join(ROOT_DIR, 'LOG','save_h5_pointnet_log.txt'), 'a')
def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


sample_cnt = 0

split_filelists = dict()
if args.split_file == '':
    split_filelists[args.split] = ['%s/%s' % (args.split, filename) for filename in os.listdir(os.path.join(ROOT_DIR, 'data', args.data_dir, args.split))]
else:
    split_filelists[args.split] = [os.path.join(args.split,args.split_file)]
    print(split_filelists[args.split][0])
    
    
for i in range(len(split_filelists[args.split])):
    filepath = os.path.join(ROOT_DIR, 'data', args.data_dir, split_filelists[args.split][i])
    log_string("input file: " + filepath)
    data, labels = common.scene2blocks_wrapper(filepath, NUM_POINT, block_size=args.block_size, stride=args.stride, random_sample=False, sample_num=None)
    log_string("output file size: " + str(data.shape) + ', ' + str(labels.shape))

    sample_cnt += data.shape[0]
    log_string("sample number now: {0}".format(sample_cnt)+"now insert_batch")
    insert_batch(data, labels, i == len(split_filelists[args.split])-1)
    log_string("finish {0} times".format(i))

print("Total test samples: {0}".format(sample_cnt))