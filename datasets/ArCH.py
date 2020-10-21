#
#
#      0=================================0
#      |    Project Name                 |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Implements read ArchDatase
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO - 2020/10/2
#
#
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import os.path
import glob
import torch
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from pc_utils import translate_pointcloud, jitter_pointcloud, rotate_pointcloud
from pc_utils import load_h5
import common
DATA_DIR = os.path.join(ROOT_DIR, 'data')


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


    def __getitem__(self, index):
        log_string("check all files:" + str(self.path_txt_all))
        filename = os.path.join(self.root, self.path_txt_all[index])
        #point_set, label = common.scenetoblocks_wrapper_normalized(filename, num_point=2048)
        point_set, label = load_h5(filename)
        # data augument
        if self.random_translate:
            point_set = translate_pointcloud(point_set[:,0:3])
        if self.random_jitter:
            point_set = jitter_pointcloud(point_set[:,0:3])
        if self.random_rotate:
            point_set = rotate_pointcloud(point_set[:,0:3])

        #conver numpy array to pytorch Tensor
        point_set = torch.from_numpy(point_set)
        #colors = torch.from_numpy()
        label = torch.from_numpy(np.array([label]).astype(np.int8))
        label = label.squeeze(0)
        log_string("read point_set: [" + str(index) + "]")
        return point_set, label


    def __len__(self):
        return len(self.path_txt_all)


LOG_FOUT = open(os.path.join(ROOT_DIR, 'LOG','datareadlog.txt'), 'w')
def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)


if __name__ == '__main__':
    datasetname = "arch_hdf5_data"
    datapath = os.path.join(DATA_DIR, datasetname)
    split = 'train'

    if datasetname == 'arch':
        print("Segmentation task:")
        d = ArchDataset(datapath, num_points=2048, split=split, random_translate=False, random_rotate=False)
        print("datasize:", d.__len__())
        ps, label = d[0]
        print(ps.size(), ps.type(), label.size(), label.type())
