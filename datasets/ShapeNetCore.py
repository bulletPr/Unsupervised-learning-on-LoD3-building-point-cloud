"""
@Author: Yuwei Cao
@File: data_processing.py
@Time: 2020/10/4 10:31AM
"""

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

#translate original point clouds
def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1),xyz2).astype('float32')
    return translated_pointcloud

#jitter point clouds operation
def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N,C = pointcloud.shape
    pointcloud += np.clip(sigma*np.random.randn(N,C),-1*clip, clip)
    return pointcloud

#rotate point clouds operation
def rotate_pointcloud(pointcloud):
    theta = np.pi*2*np.random.choice(24)/24
    rotate_matrix=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    #random rotate(x,z)
    pointcloud[:,[0,2]]=pointcloud[:,[0,2]].dot(rotate_matrix)
    return pointcloud

#shapeNet and ModelNet dataset
class Dataset(data.Dataset):
    def __init__(self, root, dataset_name='modelnet40', num_points=2048,
            split='train', load_name=False, random_rotate=False, random_jitter=False,
            random_translate=False):
        assert dataset_name.lower() in ['shapenetcorev2', 'shapenetpart', 'modelnet40',
                'modelnet10', 'arch']
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
        
        self.data = np.concatenate(data, axis=0)
        self.label = np.concatenate(label, axis=0)

    def load_path(self, type):
        path_h5py = os.path.join(self.root, '*%s*.h5'%type)
        self.path_h5py_all += glob(path_h5py)
        if self.load_name:
            path_join = os.path.join(self.root, '%s*_id2name.json'%type)
            self.path_json_all += glob(path_join)
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
            point_set = rotate_pointcloud(point_set)
        if self.random_jitter_pointcloud:
            point_set = jitter_pointcloud(point_set)
        if self.random_translate_pointcloud:
            point_set = translate_pointcloud(point_set)

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
        d = Dataset(DATA_DIR, dataset_name=datasetname, num_points=2048, random_translate=False, random_rotate=False)
        print("datasize:", d.__len__())
        ps, label = d[-1]
        print(ps.size(), ps.type(), label.size(), label.type())
