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
        with open(os.path.join(os.getcwd(), 'synsetoffset2category.txt'),'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat2id[ls[0]] = ls[1]
        self.classes = dict(zip(sorted(self.cat2id), range(len(self.cat2id))))
        log_string("classes:" + str(self.classes))

        #acquire all train/test file
        self.path_txt_all = []
        path_txt = self.root
        log_string("check path:" + str(path_txt))
        self.path_txt_all = os.listdir(path_txt)

        #load data and label
        self.path_txt_all.sort()
        log_string("check all files:" + str(self.path_txt_all))
        data, label = self.load_txt(self.path_txt_all, path_txt)

        self.data = np.concatenate(data, axis=0)
        self.label = np.concatenate(label, axis=0)
        log_string("data and label concatenated")


    #parse point cloud files
    def load_txt(self, path, root):
        all_data = []
        all_label = []
        for filename in path:
            f = root +"/"+ filename
            data = np.loadtxt(f)
            scene_points = data[:,0:3].astype('float32')
            segment_label = data[:,3].astype('int64')
            log_string(str(scene_points.shape))

            all_data.append(scene_points)
            all_label.append(segment_label)


    def __getitem__(self, index):
        point_set = self.data[index][:self.num_points]
        label = self.label[index]

        # data augument
        if self.random_translate:
            point_set = translate_pointcloud(point_set)
        if self.random_jitter:
            point_set = jitter_pointcloud(point_set)
        if self.random_rotate:
            point_set = rotate_pointcloud(point_set)

        #conver numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set)
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        label = label.squeeze(0)
        log_string("read point_set: [" + str(index) + "]")
        return(point_set, label)


    def __len__(self):
        return self.data.shape[0]


LOG_FOUT = open('log.txt', 'w')
def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


if __name__ == '__main__':
    dataset = "arch"
    datapath = "./data/arch"
    split = 'Train'

    if dataset == 'arch':
        print("Segmentation task:")
        d = ArchDataset(datapath, num_points=2048, split=split, random_translate=False, random_rotate=False)
        print("datasize:", d.__len__())
        ps, label = d[0]
        print(ps.size(), ps.type(), label.size(), label.type())
