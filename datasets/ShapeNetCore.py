#
#
#      0=================================0
#      |    Project Name                 |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Implements: knn, graph filter, foldnet/dgcnn encoder, foldnet decoder
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO - 2020/10/4 10:31AM
#
#

import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
from glob import glob
import json
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_utils


#shapeNet and ModelNet dataset
class Dataset(data.Dataset):
    def __init__(self, root, dataset_name='shapenetcorev2', num_points=2048,
            split='train', load_name=False, random_rotate=False, random_jitter=False,
            random_translate=False):
        assert dataset_name.lower() in ['shapenetcorev2', 'shapenetpart', 'modelnet40',
                'modelnet10']
        assert num_points <= 2048

        if dataset_name in ['shapenetpart','shapenetcorev2']:
            assert split.lower() in ['train','test','val','trainval','all']
        else:
            assert split.lower() in ['train','test','all']

        self.root = os.path.join(root, dataset_name + '*hdf5_2048')
        self.dataset_name = dataset_name
        self.num_points = num_points
        self.split = split
        self.load_name = load_name
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.random_translate = random_translate

        self.path_h5py_all = []
        self.path_json_all = []
        
        if self.split in ['train', 'trainval','all']:
            self.get_path('train')
        if self.dataset_name in ['shapenetpart','shapenetcorev2']:
            if self.split in ['val','trainval','all']:
                self.get_path('val')
        if self.split in ['test', 'all']:
            self.get_path('test')
        
        self.path_h5py_all.sort()
        data, label = self.load_h5py(self.path_h5py_all)
        
        if self.load_name:
            self.path_json_all.sort()
            self.name = self.load_json(self.path_json_all)    # load label name

        self.data = np.concatenate(data, axis=0)
        self.label = np.concatenate(label, axis=0)
        #this is for test
        #self.data = self.data[:100, ...]
        #self.label = self.label[:100, ...]


    def get_path(self, type):
        path_h5py = os.path.join(self.root, '*%s*.h5'%type)
        self.path_h5py_all += glob(path_h5py)
        if self.load_name:
            path_json = os.path.join(self.root, '%s*_id2name.json'%type)
            self.path_json_all += glob(path_json)
        return

    def load_h5py(self, path):
        all_data = []
        all_label = []
        for h5_name in path:
            f = h5py.File(h5_name, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
        return all_data,all_label

    def load_json(self, path):
        all_data = []
        for json_name in path:
            j = open(json_name, 'r+')
            data = json.load(j)
            all_data += data
        return all_data

    def __getitem__(self, item):
        point_set = self.data[item][:self.num_points]
        label = self.label[item]
        if self.load_name:
            name = self.name[item]
        if self.random_rotate:
            point_set = pc_utils.rotate_pointcloud(point_set)
        if self.random_jitter:
            point_set = pc_utils.jitter_pointcloud(point_set)
        if self.random_translate:
            point_set = pc_utils.translate_pointcloud(point_set)

        point_set = torch.from_numpy(point_set)
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        label = label.squeeze(0)

        if self.load_name:
            return point_set, label, name
        else:
            return point_set, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    datasetname = "shapenetcorev2"

    if datasetname == 'shapenetcorev2':
        print("Segmentation task:")
        d = Dataset(DATA_DIR, dataset_name=datasetname, num_points=2048)
        print("datasize:", d.__len__())
        ps, label = d[-1]
        print(ps.size(), ps.type(), label.size(), label.type())