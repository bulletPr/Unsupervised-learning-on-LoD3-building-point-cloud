
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
import os.path

def get_dataloader(root,split='train', batch_size=32, 
        num_points=2048, num_workers=4, shuffle=True):
    dataset = ArchDataset(
            root=root,
            split=split,
            num_points=num_points)

    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers)

    return dataloader

if __name__ == '__main__':
    datasetname = 'arch_hdf5_data'
    dataroot = os.path.join('../data', datasetname)
    dataloader = get_dataloader(dataroot, batch_size=4, num_points=2048)
    print("dataloader size: ", dataloader.dataset.__len__())
    for iter, (pts,seg) in enumerate(dataloader):
        print("points: ", pts.shape, pts.type)
        print("segs: ", seg.shape, seg.type)
        break
