#
#
#      0=================================0
#      |    Project Name                 |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Implements: Trainer  __init__(), train()
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO - 2020/11/20 13:46 AM
#
#


# ----------------------------------------
# Import packages and constant
# ----------------------------------------
import time
import os
import sys
import numpy as np
import shutil
import torch
#import argparse
import torch.optim as optim

from tensorboardX import SummaryWriter
from model import DGCNN_FoldNet

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

sys.path.append(os.path.join(ROOT_DIR, 'datasets'))
#from ArCH import ArchDataset
from dataloader import get_dataloader
from shapenet_dataloader import get_shapenet_dataloader

sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from pc_utils import is_h5_list, load_seg_list
from net_utils import Logger

DATA_DIR = os.path.join(ROOT_DIR, 'data')


# ----------------------------------------
# Trainer class
# ----------------------------------------

class Train_Cls(object):