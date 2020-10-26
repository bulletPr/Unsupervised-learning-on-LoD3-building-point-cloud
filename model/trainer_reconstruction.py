
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
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.data_dir = os.path.join(ROOT_DIR, 'data', self.dataset_name)
        self.snapshot_interval = args.snapshot_interval
        self.gpu_mode = args.gpu_mode
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
        sys.stdout = Logger(os.path.join(snapshor_root, 'network_log.txt'))
        self.write = SummaryWriter(log_dir = self.tboard_dir)
        print(str(args))

        # load dataset by dataloader
        self.train_loader = get_dataloader(root=self.data_dir, batch_size=args.batch_size, num_workers=args.workers)
        print("training set size: ", self.train_loader.dataset.__len__())

        self.test_loader = get_dataloader(root=self.data_dir, split='Test', batch_size=args.batch_size, num_workers=args.workers)
        print("testing set size: ", self.test_loader.dataset.__len__())

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
       self.train_hist={
               'loss': [],
               'per_epoch_time': [],
               'total_time': []
        }

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
        num_batch int(len(self.train_loader.dataset)/self.batch_size)
        for iter, (pts, _) in enumerate(self.train_loader):
            if self.gpu_mode:
                pts = pts.cuda()

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
