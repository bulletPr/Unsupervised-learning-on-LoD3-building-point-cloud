
#
#
#      0=================================0
#      |    Project Name                 |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Implements
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO - 2020/10/21 5:26 PM 
#
#
import torch
from ArCH import ArchDataset
from ShapeNetCore import Dataset
import os.path
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')

def get_dataloader(root, dataset_name, split='train', batch_size=32,  
        num_points=2048, num_workers=4, shuffle=True):
    dataset = ArchDataset(
            root=root,
            dataset_name = dataset_name,
            split=split,
            num_points=num_points)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers)

    return dataloader

if __name__ == '__main__':
    dataloader = get_dataloader(root = DATA_DIR, dataset_name='arch', batch_size=4, num_points=2048)
    print("dataloader size: ", dataloader.dataset.__len__())
    for iter, (pts,seg) in enumerate(dataloader):
        print("points: ", pts.shape, pts.type)
        print("segs: ", seg.shape, seg.type)
        break
