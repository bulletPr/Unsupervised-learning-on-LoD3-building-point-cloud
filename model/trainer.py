
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
import argparse
import torch.optim as optim

from tensorboardX import SummaryWriter
from model import DGCNN_FoldNet

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

sys.path.append(os.path.join(ROOT_DIR, 'datasets'))
from ArCH import ArchDataset
from dataloader import get_dataloader

sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from pc_utils import is_h5_list, load_seg_list
from net_utils import Logger

DATA_DIR = os.path.join(ROOT_DIR, 'data')


# ----------------------------------------
# Trainer class
# ----------------------------------------

class Trainer(object):
    def __init__(self, args):
        self.dataset_name = args.dataset
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        #self.data_dir = os.path.join(ROOT_DIR, 'data')
        self.snapshot_interval = args.snapshot_interval
        self.gpu_mode = args.gpu_mode
        self.model_path = args.model_path
        self.split = args.split

        # create output directory and files
        file = [f for f in args.model_path.split('/')]
        if args.experiment_name != None:
            self.experiment_id = 'Reconstruct_' + args.experiment_name
        #elif file[-2] == 'models':
            #self.experiment_id = file[-3]
        else:
            self.experiment_id = "Reconstruct" + time.strftime('%m%d%H%M%S')
        snapshot_root = 'snapshort/%s' %self.experiment_id
        tensorboard_root = 'tensorboard/%s' %self.experiment_id
        self.save_dir = os.path.join(snapshot_root, 'models/')
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
        
        # initial dataset filelist
        print('-Preparing dataset file list...')
        if self.split == 'train':
            self.filelist = os.path.join(DATA_DIR, self.dataset_name, "train_data_files.txt")
        else:
            self.filelist = os.path.join(DATA_DIR, self.dataset_name, "test_data_files.txt")
        is_list_of_h5_list = not is_h5_list(self.filelist)
        if is_list_of_h5_list:
            seg_list = load_seg_list(self.filelist)
            print("segmentation files:" + str(len(seg_list)))
            seg_list_idx = 0
            filepath = seg_list[seg_list_idx]
            seg_list_idx = seg_list_idx + 1
        else:
            filepath = filelist
        
        # initial dataset by dataloader
        print('-Preparing dataset...')
        self.train_loader = get_dataloader(filelist=filepath, batch_size=args.batch_size, num_workers=args.workers, group_shuffle=True)
        num_train = len(self.train_loader.dataset)
        print("training set size: ", self.train_loader.dataset.__len__())
        print("validate set size: " + str(num_train))
        
        #self.test_loader = get_dataloader(filelist=filepath, batch_size=args.batch_size, num_workers=args.workers)
        #print("testing set size: ", self.test_loader.dataset.__len__())

        self.model = DGCNN_FoldNet(args)

        # load model to gpu
        if self.gpu_mode:
            self.model = self.model.cuda()

        #load pretrained model
        if args.model_path != '':
            self._load_pretrain(args.model_path)

        # initialize optimizer
        self.parameter = self.model.parameters()
        self.optimizer = optim.Adam(self.parameter, lr=0.00001*16/args.batch_size, betas=(0.9, 0.999), weight_decay=1e-6)


    def train(self):
        self.train_hist={
               'loss': [],
               'per_epoch_time': [],
               'total_time': []}
        
        best_loss = 1000000000

        # start epoch index
        if self.model_path != '':
            start_epoch = self.model_path[-7:-4]
            if start_epoch[0] == '_':
                start_epoch = start_epoch[1:]
            start_epoch=int(start_epoch)
        else:
            start_epoch = 0

        # start training
        print('training start!!!')
        start_time = time.time()
        self.model.train()
        for epoch in range(start_epoch, self.epochs):
            loss = self.train_epoch(epoch)

            # save snapeshot
            if (epoch+1) % self.snapshot_interval == 0 or epoch == 0:
                self._snapshot(epoch+1)
                if loss < best_loss:
                    best_loss = loss
                    self._snapshot('best')

            # save tensorboard
            if self.writer:
                self.writer.add_scalar('Train Loss', self.train_hist['loss'][-1], epoch)
                self.writer.add_scalar('Learning Rate', self._get_lr(), epoch)

        # finish all epochs
        self._snapshot(epoch+1)
        if loss < best_loss:
            best_loss = loss
            self._snapshot('best')
        self.train_hist['total_time'].append(time.time()-start_time)
        print("Avg one epoch time: %.2f, total %d epoches time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
            self.epoch+1, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")


    def train_epoch(self, epoch):
        epoch_start_time = time.time()
        loss_buf = []
        num_train = len(self.train_loader.dataset)
        num_batch = int(num_train/self.batch_size)
        for iter, (pts, _) in enumerate(self.train_loader):
            if self.gpu_mode:
                pts = pts.cuda()
            
            start_idx = (self.batch_size * iter) % num_train
            end_idx = min(start_idx + self.batch_size, num_train)
            batch_size_train = end_idx - start_idx
            
            if start_idx + batch_size_train == num_train:
                if is_list_of_h5_list:
                    filelist_train_prev = seg_list[(seg_list_idx - 1) % len(seg_list)]
                    filelist_train = seg_list[seg_list_idx % len(seg_list)]
                    if filelist_train != filelist_train_prev:
                        self.train_loader = get_dataloader(filelist=filepath, batch_size=args.batch_size, num_workers=args.workers, )
                        num_train = len(self.train_loader.dataset)
                    seg_list_idx = seg_list_idx + 1
            
            # forward
            self.optimizer.zero_grad()
            output, _ = self.model(pts)
            loss = self.model.get_loss(pts, output)
            # backward
            loss.backward()
            self.optimizer.zero_grad()
            loss_buf.append(loss.detach().cpu().numpy())

        # finish one epoch
        epoch_time = time.time() - epoch_start_time
        self.train_hist['per_epoch_time'].append(epoch_time)
        self.train_hist['loss'].append(np.mean(loss_buf))
        print(f'Epoch {epoch+1}: Loss {np.mean(loss_buf)}, time {epoch_time:.4f}s')
        return np.mean(loss_buf)


    def _snapshot(self, epoch):
        save_dir = os.path.join(self.save_dir, self.dataset_name)
        torch.save(self.model.state_dict(), save_dir+"_"+str(epoch)+'.pkl')
        print(f"Save model to {save_dir}_{str(epoch)}.pkl")


    def _load_pretrain(self, pretrain):
        state_dict = torch.load(pretrain, map_location='cpu')
        self.model.load_state_dict(state_dict)
        print(f"Load model from {pretrain}.pkl")


    def _get_lr(self, group=0):
        return self.optimizer.param_groups[group]['lr']
