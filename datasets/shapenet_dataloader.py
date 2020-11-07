
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
from ShapeNetCore import Dataset
import os.path
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')

sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from pc_utils import is_h5_list, load_seg_list

def get_shapenet_dataloader(root, dataset_name, split='train', batch_size=32,  
        num_points=2048, num_workers=4, shuffle=True):
    dataset = Dataset(
            root=root,
            split=split,
            dataset_name = dataset_name,
            num_points=num_points)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers)

    return dataloader
'''
if __name__ == '__main__':
    
    dataloader = get_shapenet_dataloader(root=DATA_DIR,
            dataset_name = 'shapenetcorev2', batch_size=4, num_points=2048,shuffle=True)
    print("dataloader size: ", dataloader.dataset.__len__())
    for iter, (pts,seg) in enumerate(dataloader):
        print("points: ", pts.shape, pts.type)
        print("segs: ", seg.shape, seg.type)
        break
'''