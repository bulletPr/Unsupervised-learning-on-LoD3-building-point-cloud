#
#
#      0=================================0
#      |    Project Name                 |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Implements: load output latent features from .h5 files
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO - 2020/11/20 14:33 PM 
#
#


# ----------------------------------------
# import packages
# ----------------------------------------

import os
import sys
import numpy as np
import h5py

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_main_path=os.path.abspath(os.path.join(ROOT_DIR, '../cache/'))


class Saved_latent_caps_loader(object):
    def __init__(self, feature_dir , batch_size=32, npoints=2048, with_seg=False, shuffle=True, train=False, percentage=100, resized=False):
        self.feature_dir = feature_dir
        self.percentage = percentage
        if(with_seg):
            if train:
                self.h5_file=os.path.join(dataset_main_path,self.feature_dir,"saved_train_with_sem_label.h5")
            else:
                self.h5_file=os.path.join(dataset_main_path,self.feature_dir,"saved_test_with_sem_label.h5")
        else:
            if train:
                self.h5_file=os.path.join(dataset_main_path,self.feature_dir,"saved_train_with_part_label.h5")
            else:
                self.h5_file=os.path.join(dataset_main_path,self.feature_dir,"saved_test_with_part_label.h5")

        if(resized):
            self.h5_file=self.h5_file+'_%s_resized.h5'%self.percentage
             
            
        self.batch_size = batch_size
        self.npoints = npoints
        self.shuffle = shuffle
        self.with_seg = with_seg

        self.reset()
    def reset(self):
#        ''' reset order of h5 files '''
#        self.file_idxs = np.arange(0, len(self.h5_files))
#        if self.shuffle:
#            np.random.shuffle(self.file_idxs)
        self.current_data = None
        self.current_sem_label = None
        self.current_part_label = None
#        self.current_file_idx = 0
        self.batch_idx = 0


#    def _get_data_filename(self):
#        return self.h5_files[self.file_idxs[self.current_file_idx]]

    def _load_data_file(self, filename):
        if self.with_seg:
            self.current_data, self.current_seg_label = self.load_h5(filename)
            self.batch_idx = 0                 
            if self.shuffle:
                self.current_data, self.current_seg_label, _ = self.shuffle_data(
                    self.current_data, self.current_seg_label)
        else:           
            self.current_data, self.current_part_label= self.load_h5(filename)
            self.batch_idx = 0                 
            if self.shuffle:
                self.current_data, self.current_part_label, _ = self.shuffle_data(
                    self.current_data, self.current_part_label)

    def _has_next_batch_in_file(self):
        return self.batch_idx*self.batch_size < self.current_data.shape[0]

    def num_channel(self):
        return 3

    def has_next_batch(self):
        # TODO: add backend thread to load data
        if (self.current_data is None ):
#            if self.current_file_idx >= len(self.h5_files):
#                return False
            self._load_data_file(self.h5_file)
            self.batch_idx = 0
#            self.current_file_idx += 1
        return self._has_next_batch_in_file()

    def next_batch(self):
        ''' returned dimension may be smaller than self.batch_size '''
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx+1) * self.batch_size, self.current_data.shape[0])

        data_batch = self.current_data[start_idx:end_idx, 0:self.npoints, :].copy()
        self.batch_idx += 1
        if self.with_seg:
           seg_label_batch = self.current_seg_label[start_idx:end_idx].copy()
           return data_batch, seg_label_batch
        else:
           part_label_batch = self.current_part_label[start_idx:end_idx].copy()
           return data_batch, part_label_batch

    
    def shuffle_data(self, data, seg_labels):
        """ Shuffle data and labels.
            Input:
              data: B,N,... numpy array
              label: B,... numpy array
            Return:
              shuffled data, label and shuffle indices
        """
        idx = np.arange(len(seg_labels))
        np.random.shuffle(idx)
        return data[idx, ...], seg_labels[idx, ...], idx

    
    
    def load_h5(self, h5_filename):
        f = h5py.File(h5_filename)
        data = f['data'][:]
        seg_label = f['label_seg'][:]
        
        return (data, seg_label)

if __name__ == '__main__':
    
    d = Saved_latent_caps_loader('/home/zhao/Code/dataset/pointnet_data/modelnet40_ply_hdf5_2048/')
    print(d.shuffle)
    print(d.has_next_batch())
    ps_batch, seg_label_batch = d.next_batch(True)
    print(ps_batch.shape)
    print(seg_label_batch.shape)