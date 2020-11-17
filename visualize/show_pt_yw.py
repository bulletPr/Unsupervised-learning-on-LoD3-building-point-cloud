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

def calc_4nn_cur(P, nh, nw, plot=False):
    sk=1
    jj, ii = np.meshgrid(range(nh), range(nw))
    iii = ii[sk:-sk, sk:-sk]
    jjj = jj[sk:-sk, sk:-sk]
    i = np.ravel_multi_index((iii.reshape(-1),jjj.reshape(-1)),dims=(nh,nw))
    il= i-sk
    ir= i+sk
    iu= i-nw*sk
    id= i+nw*sk

    vr = P[ir,:] - P[i,:]
    vu = P[iu,:] - P[i,:]
    N = np.stack([np.cross(vr[k,:],vu[k,:]) for k in range(vr.shape[0])])
    Nn= np.sqrt(np.sum(N**2, axis=1))
    Nn= Nn[...,np.newaxis]
    N = N/Nn #normalize
    No = np.zeros(P.shape)
    No[i,:]=N
    N = No

    dl= 1 - np.sum(N[i,:]*N[il,:], axis=1)
    dr= 1 - np.sum(N[i,:]*N[ir,:], axis=1)
    du= 1 - np.sum(N[i,:]*N[iu,:], axis=1)
    dd= 1 - np.sum(N[i,:]*N[id,:], axis=1)

    ret = np.zeros(ii.shape, dtype=np.float32)+2
    ret[sk:-sk,sk:-sk] = np.max(np.stack((dl,dr,du,dd)), axis=0).reshape(nh-sk*2, nw-sk*2)

    if plot:
        #manual color the boundaries
        ret[0:sk+1,:] = 1.75 #red
        ret[-1-sk:,:] = 0.25 #blue
        ret[:,0:sk+1] = 1.0 #green
        ret[:,-1-sk:] = 1.28 #yellow

        ax, _ = vis.draw_pts(P, ret.reshape(-1),'jet')
        iii = ii[0,:]
        jjj = jj[0,:]
        i = np.ravel_multi_index((iii, jjj), dims=(nh, nw))
        ax.plot(P[i,0],P[i,1],P[i,2],zdir='y',color='red',linewidth=2)

        iii = ii[-1,:]
        jjj = jj[-1,:]
        i = np.ravel_multi_index((iii, jjj), dims=(nh, nw))
        ax.plot(P[i,0],P[i,1],P[i,2],zdir='y',color='blue',linewidth=2)

        iii = ii[:,0]
        jjj = jj[:,0]
        i = np.ravel_multi_index((iii, jjj), dims=(nh, nw))
        ax.plot(P[i,0],P[i,1],P[i,2],zdir='y',color='green',linewidth=2)

        iii = ii[:,-1]
        jjj = jj[:,-1]
        i = np.ravel_multi_index((iii, jjj), dims=(nh, nw))
        ax.plot(P[i,0],P[i,1],P[i,2],zdir='y',color='yellow',linewidth=2)

        vis.plt.figure(); vis.plt.imshow(ret,cmap='jet'); vis.plt.colorbar()
        vis.plt.title('Fold'); vis.plt.gca().set_axis_off()
    return ret

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='foldnet', metavar='N',
                        choices=['foldnet', 'dgcnn_cls', 'dgcnn_seg'],
                        help='Encoder to use, [foldnet, dgcnn_cls, dgcnn_seg]')
    parser.add_argument('--model', type=str, default = '',  help='model path')
    parser.add_argument('--k', type=int, default=None, metavar='N',
                        help='Num of nearest neighbors to use for KNN')
    parser.add_argument('--feat_dims', type=int, default=512, metavar='N',
                        help='Number of dims for feature ')
    args = parser.parse_args()
    print (args)
    
    np.random.seed(100)
    pt = np.random.rand(250,3)
    # fig = plt.figure()
    # ax = fig.add_subplot(111,projection='3d')

    #ax.scatter(pt[:,0],pt[:,1],pt[:,2])
    #plt.show()


    dataloader = get_shapenet_dataloader(root=DATA_DIR,
            dataset_name = 'shapenetcorev2', split='test', batch_size=16, num_points=2048,shuffle=False)
    
    li = list(enumerate(dataloader))
    print(len(li))

    #ps,cls = shapenet_dataset[0]
    #print('ps.size:',ps.size())
    #print('ps.type:',ps.type())
    #print('cls.size',cls.size())
    #print('cls.type',cls.type())

    # ps2,cls2 = shapenet_dataset[1]

    # ax.scatter(ps[:,0],ps[:,1],ps[:,2])
    # ax.set_xlabel('X label')
    # ax.set_ylabel('Y label')
    # ax.set_zlabel('Z label')

    # # fig2 = plt.figure()
    # # a2 = fig2.add_subplot(111,projection='3d')
    # # a2.scatter(ps2[:,0],ps2[:,1],ps2[:,2])

    # plt.show()

    foldingnet = DGCNN_FoldNet(args)

    #foldingnet.load_state_dict(torch.load('cls/foldingnet_model_150.pth'))
    foldingnet.load_state_dict(torch.load(args.model))
    foldingnet.cuda()

    chamferloss = ChamferLoss()
    chamferloss = chamferloss.cuda()
    #print(foldingnet)

    foldingnet.eval()

    try:
        os.makedirs('show_output')
    except OSError:
        pass

    for i,data in enumerate(dataloader):
        points, target = data
        #points = points.transpose(2,1)
        points = points.cuda()
        recon_pc, code = foldingnet(points)
        points_show = points.cpu().detach().numpy()
        #print(points_show.shape)
        #plot and save original images
        fig_ori = plt.figure()
        a1 = fig_ori.add_subplot(111,projection='3d')
        a1.scatter(points_show[0,0,:],points_show[0,1,:],points_show[0,2,:],marker='.',s=20,c='#B8B8B8')
        #plt.show()
        plt.savefig('show_output/rec_%d.png'%i)
        
        re_show = recon_pc.cpu().detach().numpy()
        #plot and save reconstruct images
        fig_re = plt.figure()
        a2 = fig_re.add_subplot(111,projection='3d')
        cm = plt.get_cmap('jet')
        col = [cm(float(i)/(re_show.shape[-1])) for i in range(re_show.shape[-1])]
        a2.scatter(re_show[0,0,:],re_show[0,1,:],re_show[0,2,:],c=col, marker='.', s=20)
        plt.savefig('show_output/rec_%d.png'%i)
        
        points_show = points_show.transpose(0,2,1)
        re_show = re_show.transpose(0,2,1)

        np.savetxt('show_output/ori_%d.pts'%i,points_show[0])
        np.savetxt('show_output/rec_%d.pts'%i,re_show[0])

        code_save = code.cpu().detach().numpy().astype(int)
        np.savetxt('show_output/%d.bin'%i, code_save)
        if i==3:
            break