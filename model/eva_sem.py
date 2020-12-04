#
#
#      0=================================0
#      |    Project Name                 |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Implements: save output latent features to .h5 files to train semantic segmentation network
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO - 2020/11/22 21:39 PM 
#
#


# ----------------------------------------
# import packages
# ----------------------------------------
import argparse
import torch
import torch.nn.parallel
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys
import os
import numpy as np
import statistics
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../datasets')))
import s3dis_loader
import arch_dataloader

from model import DGCNN_FoldNet
from semseg_net import SemSegNet

#import h5py
import json


def load_pretrain(model, pretrain):
        state_dict = torch.load(pretrain, map_location='cpu')
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for key, val in state_dict.items():
            if key[:6] == 'module':
                name = key[7:]  # remove 'module.'
            else:
                name = key
            new_state_dict[name] = val
        model.load_state_dict(new_state_dict)
        print(f"Load model from {pretrain}") 
        return model


def main(opt):
    USE_CUDA = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    blue = lambda x:'\033[94m' + x + '\033[0m'
    if opt.dataset == 's3dis':
        cat_no = {'ceiling': 0,'floor': 1,'wall': 2,'beam': 3,'column': 4,'window': 5,'door': 6,'table': 7,
        'chair':8,'sofa':9,'bookcase':10,'board':11,'clutter':12}
    elif opt.dataset == 'arch':
        cat_no={"arch":0, "column":1, "moldings":2, "floor":3, "door_window":4, "wall":5, "stairs":6, "vault":7, "roof":8, "other":9}    
    
    #generate part label one-hot correspondence from the catagory:
    if opt.dataset == 's3dis':
        classes = ['ceiling','floor','wall','beam','column','window','door','table','chair','sofa','bookcase','board','clutter']
        class2label = {cls: i for i,cls in enumerate(classes)}
        seg_classes = class2label
        seg_label_to_cat = {}
        for i,cat in enumerate(seg_classes.keys()):
            seg_label_to_cat[i] = cat

    #colors = plt.cm.tab10((np.arange(10)).astype(int))
    #blue = lambda x:'\033[94m' + x + '\033[0m'

# load the model for point auto encoder    
    ae_net = DGCNN_FoldNet(opt)
    if opt.model != '':
        ae_net = load_pretrain(ae_net, os.path.join(ROOT_DIR, opt.model))
    if USE_CUDA:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        ae_net = ae_net.cuda()
    ae_net=ae_net.eval()
 
    
# load the model for capsule wised part segmentation      
    sem_seg_net = SemSegNet(num_class=opt.n_classes, with_rgb=False)    
    if opt.seg_model != '':
        sem_seg_net=load_pretrain(sem_seg_net, os.path.join(ROOT_DIR,opt.model))
    if USE_CUDA:
        sem_seg_net = sem_seg_net.cuda()
    sem_seg_net = sem_seg_net.eval()    
    

    if opt.dataset=='s3dis':
        print('-Preparing Loading s3dis evaluation dataset...')
        root = '../data/stanford_indoor3d/'
        NUM_CLASSES = 13
        NUM_POINT = opt.num_points
        BATCH_SIZE = opt.batch_size
        dataset = S3DISDataset(split='test', data_root=root, num_point=NUM_POINT, rgb=False, test_area=5, block_size=1.0, sample_rate=1.0, transform=None)
        log_string("start loading test data ...")
        dataLoader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, worker_init_fn = lambda x: np.random.seed(x+int(time.time())))
        log_string("classifer set size: " + dataloader.dataset.__len__())

    elif opt.dataset == 'arch':
        print('-Preparing Loading ArCH evaluation dataset...')
        data_root = '../data/'
        NUM_CLASSES = 10
        log_string('-Now loading test ArCH dataset...')
        filelist = os.path.join(data_root, "arch_pointcnn_hdf5_2048", "test_data_files.txt")
        
        # load training data
        dataloader = arch_dataloader.get_dataloader(filelist=filelist, batch_size=opt.batch_size, 
                                                num_workers=4, group_shuffle=False,shuffle=True)
        log_string("classifer set size: " + dataloader.dataset.__len__())

    #pcd_colored = PointCloud()                   
    #pcd_ori_colored = PointCloud()        
    #rotation_angle=-np.pi/4
    #cosval = np.cos(rotation_angle)
    #sinval = np.sin(rotation_angle)           
    #flip_transforms  = [[cosval, 0, sinval,-1],[0, 1, 0,0],[-sinval, 0, cosval,0],[0, 0, 0, 1]]
    #flip_transformt  = [[cosval, 0, sinval,1],[0, 1, 0,0],[-sinval, 0, cosval,0],[0, 0, 0, 1]]

    correct_sum=0
    for batch_id, data in enumerate(dataloader):

        points, seg_label = data        
        
        if(points.size(0)<opt.batch_size):
            break

        # use the pre-trained AE to encode the point cloud into latent capsules
        points_ = Variable(points)
        #points_ = points_.transpose(2, 1)
        if USE_CUDA:
            points_ = points_.cuda()

        _, latent_caps, mid_features = ae_net(points_)
        #reconstructions=reconstructions.data.cpu()
        con_code = torch.cat([code.view(-1,args.latent_vec_size,1).repeat(1,1,args.num_points), mid_features],1)

        latent_caps=con_code.cpu().detach().numpy()        
        target = torch.from_numpy(seg_label.astype(np.int64))
        # predict the part class per capsule
        #latent_caps=latent_caps.transpose(2, 1)
        output = sem_seg_net(latent_caps)        
        output_digit = output_digit.view(-1, opt.n_classes)        
        #batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
    
        target= target.view(-1,1)[:,0]
        pred_choice = output_digit.data.cpu().max(1)[1]
        
        # calculate the accuracy with the GT
        correct = pred_choice.eq(target.data.cpu()).cpu().sum()
        correct_sum=correct_sum+correct.item()        
        print(' accuracy is: %f' %(correct_sum/float(opt.batch_size*(batch_id+1)*opt.num_points)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--latent_caps_size', type=int, default=64, help='number of latent caps')
    parser.add_argument('--latent_vec_size', type=int, default=64, help='scale of latent caps')
    parser.add_argument('--encoder', type=str, default='foldingnet', help='encoder use')
    parser.add_argument('--num_points', type=int, default=2048, help='input point set size')
    parser.add_argument('--seg_model', type=str, default='snapshot/Semantic_segmentation_arch_1024_100/arch_training_data_at_epoch136.pkl', help='model path for the pre-trained part segmentation network')
    parser.add_argument('--model', type=str, default='snapshot/Reconstruct_shapenet_foldingnet_1024/models/shapenetcorev2_best.pkl', help='model path')
    parser.add_argument('--dataset', type=str, default='arch', help='dataset: arch, shapenet_part, shapenet_core13, shapenet_core55, modelent40')
    parser.add_argument('--n_classes', type=int, default=10, help='part classes in all the catagories')
    parser.add_argument('--class_choice', type=str, default='Airplane', help='choose the class to eva')
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--feat_dims', type=int, default=1024)
    parser.add_argument('--loss', type=str, default='ChamferLoss', choices=['ChamferLoss_m','ChamferLoss'],
                        help='reconstruction loss')

    opt = parser.parse_args()
    print(opt)

    USE_CUDA = True
    main(opt)