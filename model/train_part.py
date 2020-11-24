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
#      YUWEI CAO - 2020/11/20 13:46 AM
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
import torch
#import argparse
import torch.optim as optim
import h5py
from tensorboardX import SummaryWriter

from model import DGCNN_FoldNet
from partseg_net import PartSegNet

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

sys.path.append(os.path.join(ROOT_DIR, 'datasets'))

import latent_loader

sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from pc_utils import is_h5_list, load_seg_list
from net_utils import Logger

DATA_DIR = os.path.join(ROOT_DIR, 'data')
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from net_utils import Logger


def ResizeDataset(path, percentage, n_classes, shuffle):
    #dataset_main_path=os.path.abspath(os.path.join(ROOT_DIR, 'cache'))
    ori_file_name=os.path.join(path,'saved_train_with_sem_label.h5')           
    out_file_name=ori_file_name+"_%s_resized.h5"%percentage
    if os.path.exists(out_file_name):
        os.remove(out_file_name)
    fw = h5py.File(out_file_name, 'w', libver='latest')
    dset = fw.create_dataset("data", (1,opt.num_points,opt.feature_dims,),maxshape=(None, opt.num_points, opt.feature_dims), dtype='<f4')
    dset_s = fw.create_dataset("label_seg",(1,opt.num_points,),maxshape=(None,opt.num_points,),dtype='uint8')
    dset_c = fw.create_dataset("label",(1,),maxshape=(None,),dtype='uint8')
    fw.swmr_mode = True   
    f = h5py.File(ori_file_name)
    data = f['data'][:]
    part_label = f['label_seg'][:]
    cls_label = f['label'][:]
    
    #data shuffle
    if shuffle:        
        idx = np.arange(len(cls_label))
        np.random.shuffle(idx)
        data,part_label,cls_label = data[idx, ...], part_label[idx, ...],cls_label[idx]
    
    class_dist= np.zeros(n_classes)
    for c in range(len(data)):
        class_dist[cls_label[c]]+=1
    print('Ori data to size of :', np.sum(class_dist))
    print ('class distribution of this dataset :',class_dist)
        
    class_dist_new= (percentage*class_dist/100).astype(int)
    for i in range(16):
        if class_dist_new[i]<1 :
            class_dist_new[i]=1
    class_dist_count=np.zeros(n_classes)
   
    data_count=0
    for c in range(len(data)):
        label_c=cls_label[c]
        if(class_dist_count[label_c] < class_dist_new[label_c]):
            class_dist_count[label_c]+=1
            new_shape = (data_count+1,opt.latent_caps_size,opt.latent_vec_size,)
            dset.resize(new_shape)
            dset_s.resize((data_count+1,opt.latent_caps_size,))
            dset_c.resize((data_count+1,))
            dset[data_count,:,:] = data[c]
            dset_s[data_count,:] = part_label[c]
            dset_c[data_count] = cls_label[c]
            dset.flush()
            dset_s.flush()
            dset_c.flush()
            data_count+=1
    print('Finished resizing data to size of :', np.sum(class_dist_new))
    print ('class distribution of resized dataset :',class_dist_new)
    fw.close
    
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

def _snapshot(save_dir, model, epoch):
    state_dict = model.state_dict()
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, val in state_dict.items():
        if key[:6] == 'module':
            name = key[7:]  # remove 'module.'
        else:
            name = key
        new_state_dict[name] = val
    save_dir = os.path.join(save_dir, opt.dataset)
    torch.save(new_state_dict, save_dir + "_" + str(epoch) + '.pkl')
    print(f"Save model to {save_dir}_{str(epoch)}.pkl")

def main():
    blue = lambda x:'\033[94m' + x + '\033[0m'


    experiment_id = 'part_segmentation_'+ 'opt.dataset'+'_1024'
    
    snapshot_root = 'snapshot/%s' %self.experiment_id
    tensorboard_root = 'tensorboard/%s' %self.experiment_id
    save_dir = os.path.join(ROOT_DIR, snapshot_root, 'models/')
    tboard_dir = os.path.join(ROOT_DIR, tensorboard_root)

    
    #create folder to save trained models
    if opt.model == '':
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
    sys.stdout = Logger(os.path.join(ROOT_DIR, 'LOG', 'part_seg_network_log.txt'))
    writer = SummaryWriter(log_dir = tboard_dir)

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
    objnames = [line.split()[0] for line in lines]
    on2oid = {objcats[i]:i for i in range(len(objcats))}
    fin.close()

    # load the dataset
    print('-Preparing dataset...')
    data_resized=False
    if(opt.percent_training_dataset<100):            
        ResizeDataset( percentage=opt.percent_training_dataset, n_classes=opt.n_classes,shuffle=True)
        data_resized=True
   
    train_dataset =  latent_loader.Saved_latent_caps_loader(
            dataset=opt.dataset, batch_size=opt.batch_size, npoints=opt.num_points, with_seg=True, shuffle=True, train=True,resized=data_resized)
    test_dataset =  latent_loader.Saved_latent_caps_loader(
            dataset=opt.dataset, batch_size=opt.batch_size, npoints=opt.num_points, with_seg=True, shuffle=False, train=False,resized=False)


    #initial model
    part_seg_net = PartSegNet(num_classes=opt.n_classes, with_rgb=False)
    #load pretrained model
    if opt.model != '':
        part_seg_net.load_pretrain(part_seg_net, opt.model)            
    # load model to gpu
    if USE_CUDA:
        part_seg_net = part_seg_net.cuda()       
    # initialize optimizer
    optimizer = optim.Adam(part_seg_net.parameters(), lr=0.01) 
    

# start training
    n_batch = 0
    # start epoch index
    if self.model_path != '':
        start_epoch = self.model_path[-7:-4]
        if start_epoch[0] == '_':
            start_epoch = start_epoch[1:]
        start_epoch=int(start_epoch)
    else:
        start_epoch = 0
    
    print('training start!!!')
    start_time = time.time()
    loss = []
    best_loss=0
    for epoch in range(start_epoch, opt.n_epochs):
        batch_id = 0
        part_seg_net=part_seg_net.train()
        while train_dataset.has_next_batch():
            latent_caps_, part_label, cls_label = train_dataset.next_batch()            
            
            target = torch.from_numpy(part_label.astype(np.int64))
            
            # concatnate the latent caps with the one hot part label
            latent_caps = torch.from_numpy(latent_caps_).float()
            if(latent_caps.size(0)<opt.batch_size):
                continue
            latent_caps, target = Variable(latent_caps), Variable(target)
            if USE_CUDA:
                latent_caps,target = latent_caps.cuda(), target.cuda()                            
    
# forward
            optimizer.zero_grad()
            latent_caps=latent_caps.transpose(2, 1)# consider the capsule vector size as the channel in the network
            output_digit = part_seg_net(latent_caps)
            output_digit = output_digit.view(-1, opt.n_classes)        
            #batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
    
            target= target.view(-1,1)[:,0] 
            train_loss = F.nll_loss(output_digit, target)
            train_loss.backward()
            optimizer.step()
            #print('bactch_no:%d/%d, train_loss: %f ' % (batch_id, len(train_dataloader)/opt.batch_size, train_loss.item()))
           
            pred_choice = output_digit.data.cpu().max(1)[1]
            correct = pred_choice.eq(target.data.cpu()).cpu().sum()
            
            batch_id+=1
            n_batch=max(batch_id,n_batch)
            print('[%d: %d/%d] %s loss: %f accuracy: %f' %(epoch, batch_id, n_batch, blue('test'), train_loss.item(), correct.item()/float(opt.batch_size * opt.latent_caps_size)))
            # save tensorboard
            total_time = time.time()-start_time 
            print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(total_time),
                                                                        epoch, total_time)
            loss.append(train_loss)
            writer.add_scalar('Train Loss', loss[-1], epoch)
        
        if epoch % 5 == 0:    
            part_seg_net = part_seg_net.eval()    
            correct_sum=0
            batch_id=0
            while test_dataset.has_next_batch():
                latent_caps, part_label, cls_label = test_dataset.next_batch()

                target = torch.from_numpy(seg_label.astype(np.int64))
        
                latent_caps = torch.from_numpy(latent_caps).float()
                if(latent_caps.size(0)<opt.batch_size):
                    continue
                latent_caps, target = Variable(latent_caps), Variable(target)   
                if USE_CUDA:
                    latent_caps,target = latent_caps.cuda(), target.cuda()
                
                latent_caps=latent_caps.transpose(2, 1)        
                output=part_seg_net(latent_caps)
                output = output.view(-1, opt.n_classes)        
                target= target.view(-1,1)[:,0] 
        
#                print('bactch_no:%d/%d, train_loss: %f ' % (batch_id, len(train_dataloader)/opt.batch_size, train_loss.item()))
               
                pred_choice = output.data.cpu().max(1)[1]
                correct = pred_choice.eq(target.data.cpu()).cpu().sum()                
                correct_sum=correct_sum+correct.item()
                batch_id+=1
            
            snapshot(part_seg_net, epoch + 1)
            if loss < best_loss:
                best_loss = loss
                snapshot(part_seg_net, 'best')

            print(' accuracy of epoch %d is: %f' %(epoch,correct_sum/float((batch_id+1)*opt.batch_size * opt.num_points)))
            dict_name=opt.outf+'/'+ str(opt.latent_caps_size)+'caps_'+str(opt.num_points)+'vec_'+ str(opt.percent_training_dataset) + '% of_training_data_at_epoch'+str(epoch)+'.pth'
            torch.save(part_seg_net.module.state_dict(), dict_name)
             
        train_dataset.reset()
        test_dataset.reset()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=300, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')

    parser.add_argument('--prim_caps_size', type=int, default=1024, help='number of primary point caps')
    parser.add_argument('--prim_vec_size', type=int, default=16, help='scale of primary point caps')
    parser.add_argument('--latent_caps_size', type=int, default=64, help='number of latent caps')
    parser.add_argument('--latent_vec_size', type=int, default=64, help='scale of latent caps')

    parser.add_argument('--num_points', type=int, default=2048, help='input point set size')
    parser.add_argument('--model', type=str, default='../AE/tmp_checkpoints/shapenet_part_dataset__64caps_64vec_70.pth', help='model path')
    parser.add_argument('--dataset', type=str, default='shapenet_part', help='dataset: shapenet_part, shapenet_core13, shapenet_core55, modelent40')
    parser.add_argument('--outf', type=str, default='tmp_checkpoints', help='output folder')
    parser.add_argument('--percent_training_dataset', type=int, default=100, help='traing cls with percent of training_data')
    parser.add_argument('--n_classes', type=int, default=50, help='part classes in all the catagories')

    opt = parser.parse_args()
    print(opt)

    USE_CUDA = True
    main()
