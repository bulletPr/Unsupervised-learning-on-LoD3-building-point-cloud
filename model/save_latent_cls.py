#
#
#      0=================================0
#      |    Project Name                 |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------------
#
#      Implements: Used code/feature in pretained model to infer features of input and the output features are used to SVM
#
# ----------------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO - 2020/10/26 13:30 PM 
#
#


# ----------------------------------------
# Import packages and constant
# ----------------------------------------
from __future__ import print_function
import time
import os
import sys
import numpy as np
import shutil
import h5py
import torch

#from tensorboardX import SummaryWriter
from model import DGCNN_FoldNet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'datasets'))

from shapenet_dataloader import get_shapenet_dataloader
import modelnet40_loader
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from net_utils import Logger

class SaveClsFile(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.gpu_mode = args.gpu_mode
        self.dataset_name = args.svm_dataset
        self.data_dir = os.path.join(ROOT_DIR, 'data')
        self.workers = args.workers

        #create outpu directory and files
        #file = [f for f in args.model_path.split('/')]
        if args.experiment_name != None:
            self.experiment_id = args.experiment_name
        else:
            self.experiment_id = time.strftime('%m%d%H%M%S')
        cache_root = 'cache/%s' % self.experiment_id
        os.makedirs(cache_root, exist_ok=True)
        self.feature_dir = os.path.join(cache_root, 'features/')
        sys.stdout = Logger(os.path.join(cache_root, 'gen_svm_h5_log.txt'))

        #check directory
        if not os.path.exists(self.feature_dir):
            os.makedirs(self.feature_dir)
        else:
            shutil.rmtree(self.feature_dir)
            os.makedirs(self.feature_dir)

        #print args
        print(str(args))
        print('-Preparing evaluation dataset...')  
       
        elif self.dataset_name == 'shapenetcorev2':
            print('-Preparing ShapeNetCore evaluation dataset...')
            # load training data
            self.infer_loader_train = get_shapenet_dataloader(root=self.data_dir, dataset_name = self.dataset_name, split='train', num_points=args.num_points,
                    num_workers=args.num_workers, batch_size = self.batch_size)
            log_string("training set size: " + str(self.infer_loader_train.dataset.__len__()))
            
            # load testing data
            self.infer_loader_test = get_shapenet_dataloader(root=self.data_dir, dataset_name = self.dataset_name, split = 'test', num_points=args.num_points,
                    num_workers= args.num_workers, batch_size = self.batch_size)
            log_string("testing set size: "+ str(self.infer_loader_test.dataset.__len__()))

        elif self.dataset_name == 'modelnet40':
            datapath = os.path.join(self.data_dir, "modelnet40_ply_hdf5_2048")
            dataset = modelnet40_loader.ModelNetH5Dataset(root = datapath, train=True, npoints = 2048)
            self.infer_loader_train = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                          shuffle=True, num_workers=self.workers)

            test_dataset = modelnet40_loader.ModelNetH5Dataset(root = datapath, train = False, npoints = 2048)
            self.infer_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, 
                                          shuffle=True, num_workers=self.workers)

        #initialize model
        self.model = DGCNN_FoldNet(args)

        if args.model_path != '':
            self._load_pretrain(args.model_path)

        # load model to gpu
        if self.gpu_mode:
            self.model = self.model.cuda()


    def save_file(self):
        self.model.eval()

        # generate train set for SVM
        loss_buf = []
        feature_train = []
        lbs_train = []
        n = 0
        for iter, (pts, lbs) in enumerate(self.infer_loader_train):
            #log_string("batch idx: " + str(iter) + " for generating train set for SVM...")
            if self.gpu_mode:
                pts = pts.cuda()
                lbs = lbs.cuda()
            _, feature, _  = self.model(pts) #output of reconstruction network
            feature_train.append(feature.detach().cpu().numpy().squeeze(1))  #output feature used to train a svm classifer
            lbs_train.append(lbs.cpu().numpy().squeeze(1))
            if ((iter+1)*self.batch_size % 2048) == 0 or (iter+1)==len(self.infer_loader_train):
                feature_train = np.concatenate(feature_train, axis=0)
                lbs_train = np.concatenate(lbs_train, axis=0)
                f = h5py.File(os.path.join(self.feature_dir, 'train' + str(n) + '.h5'), 'w')
                f['data']=feature_train
                f['label']=lbs_train
                f.close()
                log_string("size of generate traing set: " + str(feature_train.shape) + " ," + str(lbs_train.shape))
                print(f"Train set {n} for SVM saved.")
                feature_train = []
                lbs_train = []
                n += 1
            #loss = self.model.get_loss(pts, output)
            #loss_buf.append(loss.detach().cpu().numpy())
        #print(f"Avg loss {np.mean(loss_buf)}.")
        print("finish generating train set for SVM.")

        # genrate test set for SVM
        loss_buf = []
        feature_test = []
        lbs_test = []
        n = 0
        for iter, (pts, lbs) in enumerate(self.infer_loader_test):
            #log_string("batch idx: " + str(iter) + " for generating test set for SVM...")
            if self.gpu_mode:
                pts = pts.cuda()
                lbs = lbs.cuda()
            _, feature,_ = self.model(pts)
            feature_test.append(feature.detach().cpu().numpy().squeeze(1))
            lbs_test.append(lbs.cpu().numpy().squeeze(1))
            if ((iter+1)*self.batch_size % 2048) == 0 or (iter+1)==len(self.infer_loader_train):
                feature_test = np.concatenate(feature_test, axis=0)
                lbs_test = np.concatenate(lbs_test, axis=0)
                f = h5py.File(os.path.join(self.feature_dir, 'test' + str(n) + '.h5'), 'w')
                f['data'] = feature_test
                f['label'] = lbs_test
                f.close()
                log_string("size of generate test set: " + str(feature_test.shape) + " ," + str(lbs_test.shape))
                print(f"Test set {n} for SVM saved.")
                feature_test = []
                lbs_test = []
                n += 1
            #loss = self.model.get_loss(pts, output)
            #loss_buf.append(loss.detach().cpu().numpy())
        #print(f"Avg loss {np.mean(loss_buf)}.")
        print("finish generating test set for SVM.")

        return self.feature_dir


    def _load_pretrain(self, pretrain):
        state_dict = torch.load(pretrain, map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key[:6] == 'module':
                name = key[7:]  # remove 'module.'
            else:
                name = key
            if key[:10] == 'classifier':
                continue
            new_state_dict[name] = val
        self.model.load_state_dict(new_state_dict)
        print(f"Load model from {pretrain}")


LOG_FOUT = open(os.path.join(ROOT_DIR, 'LOG','gen_svm_h5_log.txt'), 'w')
def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)