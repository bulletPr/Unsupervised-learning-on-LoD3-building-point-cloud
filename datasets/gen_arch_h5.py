
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_utils
import common

# Constants
data_dir = os.path.join(ROOT_DIR, 'data')
arch_data_dir = os.path.join(data_dir, 'arch', 'train')
NUM_POINT = 8192
H5_BATCH_SIZE = 1000
data_dim = [NUM_POINT, 6]
label_dim = [NUM_POINT]
data_dtype = np.float32
label_dtype = np.int8

path_txt_all = []
path_txt_all = os.listdir(arch_data_dir)


output_dir = os.path.join(data_dir, 'arch_hdf5_data', 'train1')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_filename_prefix = os.path.join(output_dir, 'h5_data_all')


# --------------------------------------
#           BATCH WRITE TO HDF5
# --------------------------------------

batch_data_dim = [H5_BATCH_SIZE] + data_dim #[1000, 2048, 9]
batch_label_dim = [H5_BATCH_SIZE] + label_dim #[1000, 2048]
h5_batch_data = np.zeros(batch_data_dim, dtype = np.float32) #(1000, 2048, 9)
h5_batch_label = np.zeros(batch_label_dim, dtype = np.uint8) #(1000, 2048)
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


LOG_FOUT = open(os.path.join(ROOT_DIR, 'LOG','savetoh5log.txt'), 'a')
def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

if __name__ == '__main__':
    sample_cnt = 0
    root = '../data/arch/'
    splits = ['train', 'test']
    split_filelists = dict()
    for split in splits:
        split_filelists[split] = ['%s/%s\n' % (split, filename) for filename in os.listdir(os.path.join(root, split))
                                  if filename.endswith('.h5')]
    train_h5 = split_filelists['train']
    random.shuffle(train_h5)
    
    sampling_strategy = 'random'
    
    for i in range(len(train_h5)):
        filepath = os.path.join(root, train_h5[i].strip())
        log_string("input file: " + filepath)
        data, labels = pc_utils.load_h5(filepath)
        print('{0}, {1}'.format(data.shape, labels.shape))
        log_string("output file size: " + str(data.shape) + ', ' + str(labels.shape))
        
        log_string("Sampling data now")

        sample_cnt += data.shape[0]
        log_string("sample number now: {0}".format(sample_cnt)+"now insert_batch")
        insert_batch(data, labels, i == len(path_txt_all)-1)
        log_string("finish {0} times".format(i))

    print("Total samples: {0}".format(sample_cnt))
