
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
#      YUWEI CAO - 2020/10/25 08:54 AM
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

from tensorboardX import SummaryWriter
from model import DGCNN_FoldNet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'datasets'))
from ArCH import ArchDataset
from dataloader import get_dataloader
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from net_utils import Logger


class Reconstruct_Trainer(object):
    def __init__(self, args):
        self.dataset_name = args.dataset
        if args.epoch != None:
            self.epoch = args.epoch
        elif args.encoder == 'foldnet':
            self.epochs = 278
        elif args.encoder == 'dgcnn_cls':
            self.epoch == 250
        elif args.encoder == 'dgcnn_seg':
            self.epoch == 290
        self.batch_size = args.batch_size
        self.data_dir = os.path.join(ROOT_DIR, 'data', self.dataset_name)
        self.snapshot_interval = args.snapshot_interval
        self.no_cuda = args.no_cuda
        self.model_path = args.model_path

        # create output directory and files
        file = [f for f in args.model_path.split('/')]
        if args.experiment_name != None:
            self.experiment_id = 'Reconstruct_' + args.experiment_name
        elif file[-2] == 'models':
            self.experiment_id = file[-3]
        else:
            self.experiment_id = "Reconstruct" + time.strftime('%m%d%H%M%S')
        snapshor_root = 'snapshort/%s' %self.experiment_id
        tensorboard_root = 'tensorboard/%s' %self.experiment_id
        self.save_dir = os.path.join(snapshor_root, 'models/')
        self.tboard_dir = tensorboard_root

        #chenck arguments
        if self.model_path == '':
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            else:
                choose = input("Remove " + self.save_dir + " ? (y/n)")
                if choose == 'y':
                    shutil.rmtree(self.save_dir)
                    os.makedirs(self.save_dir)
                else:
                    sys.exit(0)
            if not os.path.exists(self.tboard_dir):
                os.makedirs(self.tboard_dir)
            else:
                shutil.rmtree(self.tboard_dir)
                os.makedirs(self.tboard_dir)
        sys.stdout = Logger(os.path.join(ROOT_DIR, 'LOG', 'network_log.txt'))
        self.write = SummaryWriter(log_dir = self.tboard_dir)
        print(str(args))

        # load dataset by dataloader
        self.train_loader = get_dataloader(root=self.data_dir, batch_size=args.batch_size, num_workers=args.workers)
        print("training set size: ", self.train_loader.dataset.__len__())

        self.test_loader = get_dataloader(root=self.data_dir, split='Test', batch_size=args.batch_size, num_workers=args.workers)
        self.model = DGCNN_FoldNet(args)

        # load model to gpu
        if self.gpu_mode:
            self.model = self.model.cuda()

        #load pretrained model
        if args.pretrain != '':
            self._load_pretrain(args.pretrain)

        # initialize optimizer
        self.parameter = self.model.parameters()
        self.optimizer = optim.Adam(self.parameter, lr=0.00001*16/args.batch_size, betas=(0.9, 0.999), weight_decay=1e-6)


    def train(self):
        
