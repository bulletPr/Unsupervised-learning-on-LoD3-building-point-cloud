
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
#      YUWEI CAO - 2020/10/26 13:30PM 
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
import h5py

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

class Inference(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.gpu_mode = args.gpu_mode
        self.dataset_name = args.dataset
        self.data_dir = os.path.join(ROOT_DIR, 'data', self.dataset_name)

        #create outpu directory and files
        file = [f for f in args.model_path.split('/')]
        if args.experiment_name != None:
            self.experiment_id = args.experiment_name
        else:
            self.experiment_id = time.strftime('%m%d%H%M%S')
        cache_root = 'cache/%s' % self.experiment_id
        os.makedirs(cache_root, exist_ok=True)
        self.feature_dir = os.path.join(cache_root, 'features/')
        sys.stdout = Logger(os.path.join(cahce_root, 'inference_log.txt'))

        #check directory
        if not os.path.exists(self.feature_dir):
            os.makedirs(self.feature_dir)
        else:
            shutil.rmtree(self.feature_dir)
            os.makedirs(self.feature_dir)

        #print args
        print(str(args))

        self.infer_loader_train = get_dataloader(root=self.data_dir, num_points=args.num_points,
                num_workers=args.num_workers)
        self.infer_loader_test = get_dataloader(root=self.data_dir, split = 'Test', num_points=args.num_points,
                num_workers= args.num_workers)
        print("training set size: ", self.infer_train_loader.dataset.__len__())
        print("testing set size: ", self.infer_test_loader.dataset.__len__())

        #initialize model
        self.model = DGCNN_FoldingNet(args)

        if args.model_path != '':
            self._load_pretrain(args.model_path)

        # load model to gpu
        if self.gpu_mode:
            self.model = self.model.cuda()


    def evaluate(self):
        self.model.eval()

        # generate train set for SVM
        loss_buf = []
        feature_train = []
        lbs_train = []
        n = 0
        for iter, (pts, lbs) in enumerate(self.infer_loader_train):
            if self.gpu_mode:
                pts = pts.cuda()
                lbs = lbs.cuda()
            output, feature  = self.model(pts)
            feature_train.append(feature.detach().cpu().numpy().squeeze(1))
            lbs_train.append(lbs.cpu().numpy().squeeze(1))
            if ((iter+1)*self.batch_size % 2048) == 0 or (iter+1)==len(self.infer_loader_train):
                feature_train = np.concatenate(feature_train, axis=0)
                lbs_train = np.concatenate(lbs_train, axis=0)
                f = h5py.File(os.path.join(self.feature_dir, 'train' + str(n) + '.h5'), 'w')
                f['data']=feature_train
                f['label']=lbs_train
                f.close()
                print("Train set {} for SVM saved.".format(n))
                feature_train = []
                lbs_train = []
                n += 1
            loss = self.model.get_loss(pts, output)
            loss_buf.append(loss.detach().cpu.numpy())
        print(f'Avg loss {np.mean(loss_buf)}')
        print("finish generating train set for SVM.")

        # genrate test set for SVM
        loss_buf = []
        feature_test = []
        lbs_test = []
        n = 0
        for iter, (pts, lbs) in enumerate(self.infer_loader_test):
            if gpu_mode:
                pts = pts.cuda()
                lbs = lbs.cuda()
            output, feature = self.model(pts)
            feature_test.append(feature.detach().cpu().numpy().squeeze(1))
            lbs_test.append(lbs.cpu().numpy().squeeze(1))
            if ((iter+1)*self.batch_size % 2048) == 0 or (iter+1)==len(self.infer_loader_train):
                feature_test = np.concatenate(feature_test, axis=0)
                lbs_test = np.concatenate(lbs_test, axis=0)
                f = h5py.File(os.path.join(self.feature_dir, 'test' + str(n) + '.h5'), 'w')
                f['data'] = feature_test
                f['label'] = lbs_test
                f.close()
                print("Test set {} for SVM saved.".format(n))
                feature_test = []
                lbs_test = []
                n += 1
            loss = self.model.get_loss(pts, output)
            loss_buf.append(loss.detach().cpu().numpy())
        print(f'Avg loss {np.mean(loss_buf)}')
        print("finish generating train set for SVM.")

        return self.feature_dir


    def _load_pretrain(self, pretrain):
        state_dict = torch.load(pretrain, map_location='cpu')
        self.model.load_state_dict(state_dict)
        print(f"Load model from {pretrain}")
