#
#
#      0=================================0
#      |    Project Name                 |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Implements Common Functions: scene to blocks
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO  -  2020/10/20 11:08 AM
#
#
# -----------------------------------------------------------------------------
# Import packages and constants
# -----------------------------------------------------------------------------
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('../')
import numpy as np


# -----------------------------------------------------------------------------
# Split scene into blocks
# -----------------------------------------------------------------------------
def scenetoblocks(data, label, num_point, block_size=1.0, stride=1.0, sampling=False,
        sampling_num=None, sample_aug=1):
    """
    Prepare block training data.
    Args:
        data: (N,9) numpy array, 012 are XYZ in meters, 345 are RGB in [0,255]
        label: N size unin8 numpy array from 0-9
        num_point: int, how many points to sample in each block
        block size: float, pysical size of the block in meters
        stride: float, stride for block sweeping
        sampling: bool, if True, we will randomly sample blocks in the room
        sampling_type: string, random, fps or grid
        sample_num: , if sample, how many blocks to sample
            [default: building area]
        sample_aug: if random sample, how much aug
    Returns:
        block_data: K x num_point x 9 np array of XYZRGBNxNyNz
        block_labels: K x num_point x 1 np array of unit8 labels
    """
    assert(stride<=block_size)

    #get the corner location for our sampling blocks
    limit = np.amax(data,0)[0:3]

    #calculate number of blocks and add into block list
    xbeg_list = []
    ybeg_list = []
    if not sampling:
        num_block_x = int(np.ceil((limit[0] - block_size) / stride)) + 1
        num_block_y = int(np.ceil((limit[1] - block_size) / stride)) + 1
        for i in range(num_block_x):
            for j in range(num_block_y):
                xbeg_list.append(i*stride)
                ybeg_list.append(j*stride)
    else:
        num_block_x = int(np.ceil(limit[0] / block_size))
        num_block_y = int(np.ceil(limit[1] / block_size))
        if sample_num is None:
            sample_num = num_block_x * num_block_y * sample_aug
        for _ in range(sample_num):
            xbeg = np.random.uniform(-block_size, limit[0])
            ybeg = np.random.uniform(-block_size, limit[1])
            xbeg_list.append(xbeg)
            ybeg_list.append(ybeg)

    #Collect blocks
    block_data_list = []
    block_label_list = []
    idx = 0
    for idx in range(len(xbeg_list)):
       xbeg = xbeg_list[idx]
       ybeg = ybeg_list[idx]
       xcond = (data[:,0]<=xbeg+block_size) & (data[:,0]>=xbeg)
       ycond = (data[:,1]<=ybeg+block_size) & (data[:,1]>=ybeg)
       cond = xcond & ycond
       if np.sum(cond) < 100: # discard block if there are less than 100 pts.
           continue

       block_data = data[cond, :]
       block_label = label[cond]

       # randomly subsample data
       block_data_sampled, block_label_sampled = \
           sample_data_label(block_data, block_label, num_point)
       block_data_list.append(np.expand_dims(block_data_sampled, 0))
       block_label_list.append(np.expand_dims(block_label_sampled, 0))

    return np.concatenate(block_data_list, 0), \
           np.concatenate(block_label_list, 0)

def room2blocks_plus(data_label, num_point, block_size, stride,
                     random_sample, sample_num, sample_aug):
    """ room2block wit RGB preprocessing.
    """
    data = data_label[:,0:6]
    data[:,3:6] /= 255.0
    label = data_label[:,-1].astype(np.uint8)

    return room2blocks(data, label, num_point, block_size, stride,
                       sampling, sample_num, sample_aug)

def room2blocks_wrapper(data_label_filename, num_point, block_size=1.0, stride=1.0,
                        sampling=False, sample_num=None, sample_aug=1):
    if data_label_filename[-3:] == 'txt':
        data_label = np.loadtxt(data_label_filename)
    elif data_label_filename[-3:] == 'npy':
        data_label = np.load(data_label_filename)
    else:
        print('Unknown file type! exiting.')
        exit()
    return room2blocks_plus(data_label, num_point, block_size, stride,
                            sampling, sample_num, sample_aug)

def room2blocks_plus_normalized(data_label, num_point, block_size, stride,
                                sampling, sample_num, sample_aug):
    """ room2block, with RGB preprocessing.
        for each block centralize XYZ, add normalized XYZ as 678 channels
    """
    data = data_label[:,0:6]
    data[:,3:6] /= 255.0
    label = data_label[:,-1].astype(np.uint8)
    max_room_x = max(data[:,0])
    max_room_y = max(data[:,1])
    max_room_z = max(data[:,2])

    data_batch, label_batch = room2blocks(data, label, num_point, block_size, stride,
                                          sampling, sample_num, sample_aug)
    new_data_batch = np.zeros((data_batch.shape[0], num_point, 9))
    for b in range(data_batch.shape[0]):
        new_data_batch[b, :, 6] = data_batch[b, :, 0]/max_room_x
        new_data_batch[b, :, 7] = data_batch[b, :, 1]/max_room_y
        new_data_batch[b, :, 8] = data_batch[b, :, 2]/max_room_z
        minx = min(data_batch[b, :, 0])
        miny = min(data_batch[b, :, 1])
        data_batch[b, :, 0] -= (minx+block_size/2)
        data_batch[b, :, 1] -= (miny+block_size/2)
    new_data_batch[:, :, 0:6] = data_batch
    return new_data_batch, label_batch

def room2blocks_wrapper_normalized(data_label_filename, num_point, block_size=1.0, stride=1.0,
                                 sampling=False, sample_num=None, sample_aug=1):
    if data_label_filename[-3:] == 'txt':
        data_label = np.loadtxt(data_label_filename)
    elif data_label_filename[-3:] == 'npy':
        data_label = np.load(data_label_filename)
    else:
        print('Unknown file type! exiting.')
        exit()
    return room2blocks_plus_normalized(data_label, num_point, block_size, stride,
                                       sampling, sample_num, sample_aug)

