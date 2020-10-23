
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
#      YUWEI CAO - 2020/10/22 9:29 AM
#
#


# ----------------------------------------
# import packages
# ----------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
from loss import ChamferLoss


# ----------------------------------------
# KNN
# ----------------------------------------

def knn(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)

    inner = -2*torch.matmul(x.transpose(2,1),x)
    xx = torch.sum(x**2, dim=1, keepdim=True)

# ----------------------------------------
# Local Convolution
# ----------------------------------------


# ----------------------------------------
# Local Maxpool
# ----------------------------------------


# ----------------------------------------
# Encoder
# ----------------------------------------

class FoldNet_Encoder(nn.Module):
    def __init__(self, args):
        super(FoldNet_Encoder, self).__init__()
        if args.k == None:
            self.k = 16
        else:
            self.k = args.k
        self.n = 2048 # input point cloud size
        self.mlp1 = nn.Sequential(
                nn.Conv1d(9,64,1),
                nn.ReLU(),
                nn.Conv1d(64,64,1),
                nn.ReLU(),
                nn.Conv1d(64,64,1),
                nn.ReLU(),
        )

        self.linear1 = nn.Linear(64,64)
        self.conv1 = nn.Conv1d(64,128,1)
        self.linear2 = nn.Linear(128,128)
        self.conv2 = nn.Conv1d(128,1024,1)
        self.mlp2 = nn.Sequential(
               nn.Conv1d(1024, args.feat_dims, 1),
               nn.ReLU(),
               nn.Conv1d(args.feat_dims, args.feat_dims, 1),
        )




