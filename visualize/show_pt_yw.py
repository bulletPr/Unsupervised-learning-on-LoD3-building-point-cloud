import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
from PIL import Image
import os
import os.path
import errno
import torch
import argparse
import json
import codecs
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, 'data')

sys.path.append(os.path.join(ROOT_DIR, 'datasets'))
from shapenet_dataloader import get_shapenet_dataloader

sys.path.append(os.path.join(ROOT_DIR, 'model'))
from model import DGCNN_FoldNet
from loss import ChamferLoss


def load_pretrain(model, pretrain):
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
        model.load_state_dict(new_state_dict)
        print(f"Load model from {pretrain}")
        return model


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default = '',  help='model path')
    parser.add_argument('--encoder', type=str, default='foldnet', metavar='N',
                        choices=['foldnet', 'dgcnn_cls', 'dgcnn_seg'],
                        help='Encoder to use, [foldnet, dgcnn_cls, dgcnn_seg]')
    parser.add_argument('--k', type=int, default=None, metavar='N',
                        help='Num of nearest neighbors to use for KNN')
    parser.add_argument('--feat_dims', type=int, default=512, metavar='N',
                        help='Number of dims for feature ')
    args = parser.parse_args()
    print (args)
    
    #np.random.seed(100)
    #pt = np.random.rand(2500,3)

    dataloader = get_shapenet_dataloader(root=DATA_DIR,
            dataset_name = 'shapenetcorev2', split='test', batch_size=1, num_points=2048,shuffle=False)
    #li = list(enumerate(dataloader))
    #print(len(li))
    foldingnet = DGCNN_FoldNet(args)

    foldingnet = load_pretrain(foldingnet, args.model)
    foldingnet.cuda()
    foldingnet.eval()

    try:
        os.makedirs('rec_output')
    except OSError:
        pass

    for i,data in enumerate(dataloader):
        points, target = data
        points = points.cuda()
        recon_pc, _ = foldingnet(points.view(1,2048,3))
        points_show = points.cpu().detach().numpy() #(1,2048,3)

        #plot and save original images
        fig_ori = plt.figure()
        a1 = fig_ori.add_subplot(111,projection='3d')
        a1.scatter(points_show[0,:,1],points_show[0,:,1],points_show[0,:,2],marker='.',s=20,c='#B8B8B8')
        n_epoch = args.model[-7:-4]
        plt.savefig('rec_output/ori_%s_%d.png'%(n_epoch,i))
        
        re_show = recon_pc.cpu().detach().numpy()
        #plot and save reconstruct images
        fig_re = plt.figure()
        a2 = fig_re.add_subplot(111,projection='3d')
        cm = plt.get_cmap('jet')
        col = [cm(float(i)/(re_show.shape[-2])) for i in range(re_show.shape[-2])]
        a2.scatter(re_show[0,:,0],re_show[0,:,1],re_show[0,:,2],c='#B8B8B8', marker='.', s=20)
        plt.savefig('rec_output/rec_%s_%d.png'%(n_epoch,i))
        
        points_show = points_show.transpose(0,2,1)
        re_show = re_show.transpose(0,2,1)

        np.savetxt('rec_output/ori_%s_%d.pts'%(n_epoch,i),points_show[0])
        np.savetxt('rec_output/rec_%s_%d.pts'%(n_epoch,i),re_show[0])

        #code_save = code.cpu().detach().numpy().astype(int)
        #np.savetxt('show_output/%d.bin'%i, code_save)
        if i==1:
            break