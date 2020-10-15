from torch.utils.data.dataset import Dataset
import os
import numpy as np
import os.path
from glob import glob
import torch

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.rand()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud

#ArCH dataset load
class ArchDataset(Dataset):
    def __init__(self, root, num_points=2048, random_translate=False, random_rotate=False,
            random_jitter=False, split='Train'):
        self.random_translate = random_translate
        self.random_jitter = random_jitter
        self.random_rotate = random_rotate
        self.cat2id = {}
        self.root = os.path.join(root, split)

        #parse category file
        with open('synsetoffset2category.txt','r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat2id[ls[0]] = ls[1]
        self.classes = dict(zip(sorted(self.cat2id), range(len(self.cat2id))))
        f.close()
        log_string("classes:" + str(self.classes))

        #acquire all train/test file
        self.path_txt_all = []
        self.path_txt_all = os.listdir(self.root)
        log_string("check paths:" + str(self.path_txt_all))

        #save data path
        self.path_txt_all.sort()
        log_string("check sorted paths:" + str(self.path_txt_all))

    #parse point cloud files
    def load_txt(self, filename):
        all_data = []
        all_label = []
        log_string("loading data in: " + str(filename))
        data = np.loadtxt(filename)
        scene_xyz = data[:,0:3].astype(np.float32)
        points_colors = data[:,3:6].astype(np.int8)
        segment_label = data[:,6].astype(np.int8)
        log_string(str(scene_xyz.shape))

        return scene_xyz, points_colors, segment_label


    def __getitem__(self, index):
        log_string("check all files:" + str(self.path_txt_all))
        filename = os.path.join(self.root, self.path_txt_all[index])
        point_set, color, label = self.load_txt(filename)

        #self.data = np.concatenate(data, axis=0)
        #self.label = np.concatenate(label, axis=0)

        #point_set = point_set[:self.num_points]
        #label = label[]

        # data augument
        if self.random_translate:
            point_set = translate_pointcloud(point_set)
        if self.random_jitter:
            point_set = jitter_pointcloud(point_set)
        if self.random_rotate:
            point_set = rotate_pointcloud(point_set)

        #conver numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set)
        #colors = torch.from_numpy()
        label = torch.from_numpy(np.array([label]).astype(np.int8))
        label = label.squeeze(0)
        log_string("read point_set: [" + str(index) + "]")
        return point_set, label


    def __len__(self):
        return len(self.path_txt_all)


LOG_FOUT = open(os.path.join('..', 'LOG','datareadlog.txt'), 'w')
def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


if __name__ == '__main__':
    datasetname = "arch"
    datapath = os.path.join("../data", datasetname)
    split = 'Train'

    if datasetname == 'arch':
        print("Segmentation task:")
        d = ArchDataset(datapath, num_points=2048, split=split, random_translate=False, random_rotate=False)
        print("datasize:", d.__len__())
        ps, label = d[8]
        print(ps.size(), ps.type(), label.size(), label.type())
