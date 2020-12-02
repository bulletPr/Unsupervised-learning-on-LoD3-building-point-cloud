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
#      YUWEI CAO - 2020/11/19 20:26 PM 
#
#


# ----------------------------------------
# import packages
# ----------------------------------------
import argparse
import torch
from torch.autograd import Variable
import torch.optim as optim
import sys
import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../datasets')))
from s3dis_loader import S3DISDataset
from arch_dataloader import get_dataloader

#from open3d import *

from model import DGCNN_FoldNet

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


def main(args):
    USE_CUDA = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ae_net = DGCNN_FoldNet(args)
  
    if args.model != '':
        capsule_net = load_pretrain(ae_net, os.path.join(ROOT_DIR, args.model))

    if USE_CUDA:       
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        ae_net.cuda()
    
    if args.dataset=='s3dis':
        log_string('-Preparing Loading s3dis evaluation dataset...')
        classes = ['ceiling','floor','wall','beam','column','window','door','table','chair','sofa','bookcase','board','clutter']
        class2label = {cls: i for i,cls in enumerate(classes)}
        seg_classes = class2label
        seg_label_to_cat = {}
        for i,cat in enumerate(seg_classes.keys()):
            seg_label_to_cat[i] = cat
        if args.save_training:
            log_string('-Now loading s3dis training classifer dataset...')
            split='train'
        else:
            log_string('-Now loading test s3dis dataset...')
            split='test'
        
        root = '../data/stanford_indoor3d/'
        NUM_CLASSES = 13
        NUM_POINT = args.num_points
        BATCH_SIZE = args.batch_size
        dataset = S3DISDataset(split=split, data_root=root, num_point=NUM_POINT, rgb=False, test_area=5, block_size=1.0, sample_rate=1.0, transform=None)
        log_string("start loading test data ...")
        dataLoader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, worker_init_fn = lambda x: np.random.seed(x+int(time.time())))
        log_string("classifer set size: " + dataloader.dataset.__len__())

    elif args.dataset == 'arch':
        log_string('-Preparing Loading ArCH evaluation dataset...')
        data_root = os.path.join(ROOT_DIR, 'data')
        if args.save_training:
            log_string('-Now loading ArCH training classifer dataset...')
            filelist = os.path.join(data_root, 'arch_pointcnn_hdf5_2048', "train_data_files.txt")
        else:
            log_string('-Now loading test ArCH dataset...')
            filelist = os.path.join(data_root, 'arch_pointcnn_hdf5_2048', "test_data_files.txt")
        
        # load training data
        dataloader = get_dataloader(filelist=filelist, batch_size=args.batch_size, 
                                                num_workers=4, group_shuffle=False,shuffle=True)
        log_string("segmentation training dataset size: " + str(dataloader.dataset.__len__()))

        
# init saving process
    #pcd = PointCloud() 
    data_size=0
    dataset_main_path=os.path.abspath(os.path.join(ROOT_DIR, 'cache'))
    experiment_name = 'latent_' + args.pre_ae_epochs + '_' + args.dataset + '_' + str(args.latent_vec_size)
    out_file_path=os.path.join(dataset_main_path, experiment_name, 'features')
    if not os.path.exists(out_file_path):
        os.makedirs(out_file_path);   
    


#  process for 'shapenet_part' or 'shapenet_core13'
    ae_net.eval()
    n = 0
    feature_set = []
    label_set = []
    for batch_id, data in enumerate(dataloader):
    #for batch_id, data in (enumerate(dataloader), total=len(dataloader)):
        points, sem_label= data
        if(points.size(0)<args.batch_size):
            break
        points = Variable(points)
        #points = points.transpose(2, 1)
        if USE_CUDA:
            points = points.cuda()
            sem_label = sem_label.cuda()
        
        _, code, mid_features = ae_net(points)

        con_code = torch.cat([code.view(-1,args.latent_vec_size,1).repeat(1,1,args.num_points), mid_features],1)
        
        # write the output latent and cls into file
        feature_set.append(con_code.transpose(2,1).detach().cpu().numpy())
        label_set.append(sem_label.cpu().detach().numpy())

        if ((batch_id+1)*args.batch_size % 256) == 0 or (batch_id+1)==len(dataloader):
            feature_set = np.concatenate(feature_set, axis=0)
            label_set = np.concatenate(label_set, axis=0)
            if args.save_training:
                out_file_name=out_file_path+"/saved_train_with_sem_label_"+ str(n)+".h5"
            else:
                out_file_name=out_file_path+"/saved_test_with_sem_label_"+ str(n)+".h5"        
            if os.path.exists(out_file_name):
                os.remove(out_file_name)
            fw = h5py.File(out_file_name, 'w', libver='latest')
            fw['data'] = feature_set
            fw['label_seg'] = label_set
            fw.close()
            n+=1
            log_string('accumalate of batch %d, and datasize is %s, label size is: %s ' % ((batch_id), str(feature_set.shape), str(label_set.shape)))
            print('latent set {n} for segmentation saved')
            feature_set = []
            label_set = []
    print("finish generating latent set for SVM.")  

LOG_FOUT = open(os.path.join(ROOT_DIR, 'LOG','save_sem_latent_log.txt'), 'w')
def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)   
         

if __name__ == "__main__":
    import h5py
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--pre_ae_epochs', type=str, default='shapenetcorev2_best', help='choose which pretrained model to use')
    parser.add_argument('--latent_vec_size', type=int, default=1024, help='scale of latent caps')

    parser.add_argument('--num_points', type=int, default=2048, help='input point set size')
    parser.add_argument('--model', type=str, default='snapshot/Reconstruct_shapenet_foldingnet_1024/models/shapenetcorev2_best.pkl', help='model path')
    parser.add_argument('--dataset', type=str, default='arch', help='It has to be arch dataset')
    parser.add_argument('--save_training', help='save the output latent caps of training data or test data', action='store_true')
    parser.add_argument('--n_classes', type=int, default=10, help='catagories of current dataset')
    
    parser.add_argument('--encoder', type=str, default='foldingnet', help='encoder use')
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--feat_dims', type=int, default=1024)
    parser.add_argument('--loss', type=str, default='ChamferLoss', choices=['ChamferLoss_m','ChamferLoss'],
                        help='reconstruction loss')

    args = parser.parse_args()
    print(args)
    main(args)