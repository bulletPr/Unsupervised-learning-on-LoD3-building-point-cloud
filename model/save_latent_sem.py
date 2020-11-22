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
import torch.nn.parallel
from torch.autograd import Variable
import torch.optim as optim
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '../datasets')))

import shapenet_dataloader
from dataloader import get_dataloader

from open3d import *

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


def main():
    USE_CUDA = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ae_net = DGCNN_FoldNet(opt)
  
    if opt.model != '':
        capsule_net = load_pretrain(ae_net, opt.model)

    if USE_CUDA:       
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        ae_net.cuda()
    
    if opt.dataset=='s3dis':
        print('-Preparing Loading s3dis evaluation dataset...')
        if opt.save_training:
            log_string('-Now loading s3dis training classifer dataset...')
            split='train'
        else:
            log_string('-Now loading test s3dis dataset...')
            split='test'
        dataset = S3DISDataset(split=split, data_root=root, num_point=NUM_POINT, test_area=6, block_size=1.0, sample_rate=1.0, transform=None)
        log_string("start loading test data ...")
        dataLoader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, worker_init_fn = lambda x: np.random.seed(x+int(time.time())))
        log_string("classifer set size: " + dataloader.dataset.__len__())

    elif opt.dataset == 'arch':
        print('-Preparing Loading ArCH evaluation dataset...')
        if opt.save_training:
            log_string('-Now loading ArCH training classifer dataset...')
            filelist = os.path.join(self.data_dir, opt.dataset, "sem_data_files.txt")
        else:
            log_string('-Now loading test ArCH dataset...')
            filelist = os.path.join(self.data_dir, opt.dataset, "test_data_files.txt")
        
        # load training data
        dataloader = get_dataloader(filelist=filelist, batch_size=opt.batch_size, 
                                                num_workers=4, group_shuffle=False,shuffle=True)
        log_string("classifer set size: " + dataloader.dataset.__len__())

        
# init saving process
    #pcd = PointCloud() 
    data_size=0
    dataset_main_path=os.path.abspath(os.path.join(ROOT_DIR, 'cache'))
    experiment_name = opt.dataset + opt.encoder + '_' + opt.latent_vec_size + '_' + opt.n_epochs
    out_file_path=os.path.join(dataset_main_path, opt.experiment_name,'features')
    if not os.path.exists(out_file_path):
        os.makedirs(out_file_path);   
    if opt.save_training:
        out_file_name=out_file_path+"/saved_train_with_sem_label.h5"
    else:
        out_file_name=out_file_path+"/saved_test_with_sem_label.h5"        
    if os.path.exists(out_file_name):
        os.remove(out_file_name)
    fw = h5py.File(out_file_name, 'w', libver='latest')
    dset = fw.create_dataset("data", (1, opt.num_points, opt.latent_vec_size,),maxshape=(None,opt.num_points, opt.latent_vec_size,), dtype='<f4')
    dset_s = fw.create_dataset("label_seg",(1,opt.num_points,),maxshape=(None,opt.num_points,),dtype='uint8')
    dset_c = fw.create_dataset("label",(1,),maxshape=(None,),dtype='uint8')
    fw.swmr_mode = True


#  process for 'shapenet_part' or 'shapenet_core13'
    ae_net.eval()
    
    for batch_id, data in enumerate(dataloader):
        points, sem_label, cls_label= data
        if(points.size(0)<opt.batch_size):
            break
        points = Variable(points)
        #points = points.transpose(2, 1)
        if USE_CUDA:
            points = points.cuda()
            sem_label = sem_label.cuda()
            cls_label = cls_label.cuda()
        
        reconstructions, code, mid_features = ae_net(points)

        rep_code = code.view(-1,opt.latent_vec_size,1).repeat(1,1,opt.num_points)
        con_code = torch.cat([rep_code, mid_features],1)
       
        # For each resonstructed point, find the nearest point in the input pointset, 
        # use their part label to annotate the resonstructed point,
        # Then after checking which capsule reconstructed this point, use the part label to annotate this capsule
        reconstructions=reconstructions.datach().cpu()   
        points=points.datach().cpu()  
        for batch_no in range (points.size(0)):
            #pcd.points = Vector3dVector(points[batch_no,])
            #pcd_tree = KDTreeFlann(pcd)
            for point_id in range (opt.num_points):
                #[k, idx, _] = pcd_tree.search_knn_vector_3d(reconstructions[batch_no,point_id,:], 1)
                point_sem_label=sem_label[batch_no, idx]            
    
        # write the output latent caps and cls into file
        data_size=data_size+points.size(0)
        new_shape = (data_size,opt.num_points, opt.latent_vec_size, )
        dset.resize(new_shape)
        dset_s.resize((data_size,opt.num_points,))
        dset_c.resize((data_size,))
        
        code_=con_code.transpose(2,1).cpu().detach().numpy()

        target_=point_sem_label.numpy()
        dset[data_size-points.size(0):data_size,:,:] = code_
        dset_s[data_size-points.size(0):data_size] = target_
        dset_c[data_size-points.size(0):data_size] = cls_label.squeeze().numpy()
    
        dset.flush()
        dset_s.flush()
        dset_c.flush()
        print('accumalate of batch %d, and datasize is %d ' % ((batch_id), (dset.shape[0])))
           
    fw.close()   

    
         

if __name__ == "__main__":
    import h5py
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs to train for')

    parser.add_argument('--prim_caps_size', type=int, default=1024, help='number of primary point caps')
    parser.add_argument('--prim_vec_size', type=int, default=16, help='scale of primary point caps')
    parser.add_argument('--latent_vec_size', type=int, default=1024, help='scale of latent caps')

    parser.add_argument('--num_points', type=int, default=2048, help='input point set size')
    parser.add_argument('--model', type=str, default='../AE/tmp_checkpoints/shapenet_part_dataset__64caps_64vec_70.pth', help='model path')
    parser.add_argument('--dataset', type=str, default='shapenet_part', help='It has to be shapenet part')
#    parser.add_argument('--save_training', type=bool, default=True, help='save the output latent caps of training data or test data')
    parser.add_argument('--save_training', help='save the output latent caps of training data or test data', action='store_true')

    parser.add_argument('--n_classes', type=int, default=16, help='catagories of current dataset')
    parser.add_argument('--enocder', type=str, default='fodlingnet', help='encoder use')

    opt = parser.parse_args()
    print(opt)
    main()