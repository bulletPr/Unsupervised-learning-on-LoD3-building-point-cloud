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
#      YUWEI CAO - 2020/11/24 9:29 AM 
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
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../models')))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../datasets')))
import shapenetpart_loader
import matplotlib.pyplot as plt

from model import DGCNN_FoldNet
from partseg_net import PartSegNet

#import h5py
import json


from open3d import *
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


def main():
    blue = lambda x:'\033[94m' + x + '\033[0m'
    cat_no={'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Earphone': 5, 
            'Guitar': 6, 'Knife': 7, 'Lamp': 8, 'Laptop': 9, 'Motorbike': 10, 
            'Mug': 11, 'Pistol': 12, 'Rocket': 13, 'Skateboard': 14, 'Table': 15}    
    
#generate part label one-hot correspondence from the catagory:
    dataset_main_path=os.path.abspath(os.path.join(BASE_DIR, '../data'))
    oid2cpid_file_name=os.path.join(dataset_main_path, opt.dataset,'shapenetcore_partanno_segmentation_benchmark_v0/shapenet_part_overallid_to_catid_partid.json')        
    oid2cpid = json.load(open(oid2cpid_file_name, 'r'))   
    object2setofoid = {}
    for idx in range(len(oid2cpid)):
        objid, pid = oid2cpid[idx]
        if not objid in object2setofoid.keys():
            object2setofoid[objid] = []
        object2setofoid[objid].append(idx)
    
    all_obj_cat_file = os.path.join(dataset_main_path, opt.dataset, 'shapenetcore_partanno_segmentation_benchmark_v0/synsetoffset2category.txt')
    fin = open(all_obj_cat_file, 'r')
    lines = [line.rstrip() for line in fin.readlines()]
    objcats = [line.split()[1] for line in lines]
#    objnames = [line.split()[0] for line in lines]
#    on2oid = {objcats[i]:i for i in range(len(objcats))}
    fin.close()


    colors = plt.cm.tab10((np.arange(10)).astype(int))
    blue = lambda x:'\033[94m' + x + '\033[0m'

# load the model for point auto encoder    
    ae_net = DGCNN_FoldNet(opt)
    if opt.model != '':
        ae_net = load_pretrain(ae_net, opt.model)
    if USE_CUDA:
        ae_net = ae_net.cuda()
    ae_net=ae_net.eval()
 
    
# load the model for capsule wised part segmentation      
    part_seg_net = PartSegNet(num_classes=opt.n_classes, with_rgb=False)    
    if opt.seg_model != '':
        part_seg_net.load_pretrain(part_seg_net, opt.model)
    if USE_CUDA:
        part_seg_net = part_seg_net.cuda()
    part_seg_net = part_seg_net.eval()    
    

    train_dataset = shapenet_part_loader.PartDataset(classification=False, class_choice=opt.class_choice, npoints=opt.num_points, split='test')
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)        


    pcd_colored = PointCloud()                   
    pcd_ori_colored = PointCloud()        
    rotation_angle=-np.pi/4
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)           
    flip_transforms  = [[cosval, 0, sinval,-1],[0, 1, 0,0],[-sinval, 0, cosval,0],[0, 0, 0, 1]]
    flip_transformt  = [[cosval, 0, sinval,1],[0, 1, 0,0],[-sinval, 0, cosval,0],[0, 0, 0, 1]]
    
    
    correct_sum=0
    for batch_id, data in enumerate(train_dataloader):

        points, part_label, cls_label= data        
        if not (opt.class_choice==None ):
            cls_label[:]= cat_no[opt.class_choice]
    
        if(points.size(0)<opt.batch_size):
            break
        
        
        # use the pre-trained AE to encode the point cloud into latent capsules
        points_ = Variable(points)
        #points_ = points_.transpose(2, 1)
        if USE_CUDA:
            points_ = points_.cuda()
        reconstructions, latent_caps, _ = ae_net(points_)
        reconstructions=reconstructions.data.cpu()
        
        rep_code = code.view(-1,opt.latent_vec_size,1).repeat(1,1,opt.num_points)
        con_code = torch.cat([rep_code, mid_features],1)

        latent_caps=con_code.cpu().detach().numpy()
        #concatanete the latent caps with one-hot part label
      
        # predict the part class per capsule
        latent_caps=latent_caps.transpose(2, 1)
        output=part_seg_net(latent_caps)        
        for i in range (opt.batch_size):
            iou_oids = object2setofoid[objcats[cls_label[i]]]
            non_cat_labels = list(set(np.arange(50)).difference(set(iou_oids))) # there are 50 part classes in all the 16 catgories of objects
            mini = torch.min(output[i,:,:])
            output[i,:, non_cat_labels] = mini - 1000   
        pred_choice = output.data.cpu().max(2)[1]
       
        # assign predicted the capsule part label to its reconstructed point patch
        reconstructions_part_label=torch.zeros([opt.batch_size,opt.num_points],dtype=torch.int64)
        for i in range(opt.batch_size):
            for j in range(opt.num_points):
                for m in range(int(opt.num_points/opt.num_points)):
                    reconstructions_part_label[i,opt.num_points*m+j]=pred_choice[i,j]

        
        # assign the part label from the reconstructed point cloud to the input point set with NN
        pcd=pcd = PointCloud() 
        pred_ori_pointcloud_part_label=torch.zeros([opt.batch_size,opt.num_points],dtype=torch.int64)   
        for point_set_no in range (opt.batch_size):
            pcd.points = Vector3dVector(reconstructions[point_set_no,])
            pcd_tree = KDTreeFlann(pcd)
            for point_id in range (opt.num_points):
                [k, idx, _] = pcd_tree.search_knn_vector_3d(points[point_set_no,point_id,:], 10)
                local_patch_labels=reconstructions_part_label[point_set_no,idx]
                pred_ori_pointcloud_part_label[point_set_no,point_id]=statistics.median(local_patch_labels)
       
        
        # calculate the accuracy with the GT
        correct = pred_ori_pointcloud_part_label.eq(part_label.data.cpu()).cpu().sum()
        correct_sum=correct_sum+correct.item()        
        print(' accuracy is: %f' %(correct_sum/float(opt.batch_size*(batch_id+1)*opt.num_points)))

        
        
         # viz the part segmentation
        point_color=torch.zeros([opt.batch_size,opt.num_points,3])
        point_ori_color=torch.zeros([opt.batch_size,opt.num_points,3])

        for point_set_no in range (opt.batch_size):
            iou_oids = object2setofoid[objcats[cls_label[point_set_no ]]]
            for point_id in range (opt.num_points):
                part_no=pred_ori_pointcloud_part_label[point_set_no,point_id]-iou_oids[0]
                point_color[point_set_no,point_id,0]=colors[part_no,0]
                point_color[point_set_no,point_id,1]=colors[part_no,1]
                point_color[point_set_no,point_id,2]=colors[part_no,2]
                
            pcd_colored.points=Vector3dVector(points[point_set_no,])
            pcd_colored.colors=Vector3dVector(point_color[point_set_no,])

            
            for point_id in range (opt.num_points):
                part_no=part_label[point_set_no,point_id]-iou_oids[0]
                point_ori_color[point_set_no,point_id,0]=colors[part_no,0]
                point_ori_color[point_set_no,point_id,1]=colors[part_no,1]
                point_ori_color[point_set_no,point_id,2]=colors[part_no,2]
                
            pcd_ori_colored.points=Vector3dVector(points[point_set_no,])
            pcd_ori_colored.colors=Vector3dVector(point_ori_color[point_set_no,])
           
            pcd_ori_colored.transform(flip_transforms)# tansform the pcd in order to viz both point cloud
            pcd_colored.transform(flip_transformt)
            draw_geometries([pcd_ori_colored, pcd_colored])
            
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')

    parser.add_argument('--prim_caps_size', type=int, default=1024, help='number of primary point caps')
    parser.add_argument('--prim_vec_size', type=int, default=16, help='scale of primary point caps')
    parser.add_argument('--latent_caps_size', type=int, default=64, help='number of latent caps')
    parser.add_argument('--latent_vec_size', type=int, default=64, help='scale of latent caps')

    parser.add_argument('--num_points', type=int, default=2048, help='input point set size')
    parser.add_argument('--part_model', type=str, default='../../checkpoints/part_seg_1percent.pth', help='model path for the pre-trained part segmentation network')
    parser.add_argument('--model', type=str, default='../../checkpoints/shapenet_part_dataset_ae_200.pth', help='model path')
    parser.add_argument('--dataset', type=str, default='shapenet_part', help='dataset: shapenet_part, shapenet_core13, shapenet_core55, modelent40')
    parser.add_argument('--n_classes', type=int, default=50, help='part classes in all the catagories')
    parser.add_argument('--class_choice', type=str, default='Airplane', help='choose the class to eva')

    opt = parser.parse_args()
    print(opt)

    USE_CUDA = True
    main()