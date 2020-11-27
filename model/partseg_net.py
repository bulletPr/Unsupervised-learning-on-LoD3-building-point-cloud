#
#
#      0=================================0
#      |    Project Name                 |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Implements: Part Segmentation Network
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO - 2020/11/19 20:33 PM 
#
#


# ----------------------------------------
# import packages
# ----------------------------------------

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

USE_CUDA = True


# ----------------------------------------
# Part Segmentation Function
# ----------------------------------------

class PartSegNet(nn.Module):    
    def __init__(self, part_num, with_rgb=False):
        super(PartSegNet, self).__init__()
        if with_rgb:
            channel = 6
        else:
            channel = 3
        self.part_num = part_num

        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.part_num, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128) 

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.part_num), dim=-1)
        x = x.view(batchsize, n_pts, self.part_num)
        return x

    
if __name__ == '__main__':
    USE_CUDA = True