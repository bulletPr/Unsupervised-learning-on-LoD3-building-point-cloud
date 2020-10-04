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
import sys
import torchvision.transforms as transforms
import torch.utils.data as data


#translate original point clouds
def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    tanslated_pointcloud = np.add(np.multiply(pointcloud, xyz1),xyz2).astype('float32')
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

#shapeNet, ArCH and ModelNet dataset
class Dataset(data.Dataset):
    def __init__(self, root, dataset_name='modelnet40', num_points=2048,
            split='train', load_name=False, random_rotate=False, random_jitter=False,
            random_translate=False):
        assert dataset_name.lower() in ['shapenetcorev2', 'shapenetpart', 'modelnet40'，
                'modelnet10', 'arch']
        assert num_points <= 2048

        if dataset_name in ['shapenet']
