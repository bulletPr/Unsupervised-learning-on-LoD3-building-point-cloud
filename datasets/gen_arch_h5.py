
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import common
import pc_utils

# Constants
data_dir = os.path.join(ROOT_DIR, 'data')
arch_data_dir = os.path.join(data_dir, 'arch', 'Train1')
NUM_POINT = 2048
H5_BATCH_SIZE = 1000
data_dim = [NUM_POINT, 9]
label_dim = [NUM_POINT]
data_dtype = np.float32
label_dtype = np.int8

path_txt_all = []
path_txt_all = os.listdir(arch_data_dir)


output_dir = os.path.join(data_dir, 'arch_hdf5_data', 'train')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_filename_prefix = os.path.join(output_dir, 'ply_data_all')
output_building_filelist = os.path.join(output_dir, '../building_filelist.txt')
fout_building = open(output_building_filelist, 'w')


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
        insert_batch(data[capacity:, ...], label[capacity:, ...], last_batch)
    if last_batch and buffer_size > 0:
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        pc_utils.save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...], data_dtype, label_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, buffer_size))
        h5_index += 1
        buffer_size = 0
    return

sample_cnt = 0
for i, data_label_filename in enumerate(path_txt_all):
    data_label_filename = os.path.join(arch_data_dir, data_label_filename)
    print(data_label_filename)
    data, label = common.scenetoblocks_wrapper_normalized(data_label_filename, NUM_POINT, block_size=1.0,
            stride=1.0)
    print('{0}, {1}'.format(data.shape, label.shape))
    for _ in range(data.shape[0]):
        fout_building.write(os.path.basename(data_label_filename)[0:-4]+'\n')

    sample_cnt += data.shape[0]
    insert_batch(data, label, i == len(path_txt_all)-1)

fout_building.close()
print("Total samples: {0}".format(sample_cnt))
