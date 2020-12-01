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
import argparse
import torch.optim as optim
import h5py
from tensorboardX import SummaryWriter

from model import DGCNN_FoldNet
from semseg_net import SemSegNet

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

sys.path.append(os.path.join(ROOT_DIR, 'datasets'))
import latent_loader

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

    fw.swmr_mode = True   
    f = h5py.File(ori_file_name)
    data = f['data'][:]
    seg_label = f['label_seg'][:]
    
    #data shuffle
    if shuffle:        
        idx = np.arange(len(seg_label))
        np.random.shuffle(idx)
        data,seg_label = data[idx, ...], seg_label[idx, ...]
    
    class_dist= np.zeros(n_classes)
    for c in range(len(data)):
        class_dist[seg_label[c]]+=1
    print('Ori data to size of :', np.sum(class_dist))
    print ('class distribution of this dataset :',class_dist)
        
    class_dist_new= (percentage*class_dist/100).astype(int)
    for i in range(16):
        if class_dist_new[i]<1 :
            class_dist_new[i]=1
    class_dist_count=np.zeros(n_classes)
   
    data_count=0
    for c in range(len(data)):
        label_c=seg_label[c]
        if(class_dist_count[label_c] < class_dist_new[label_c]):
            class_dist_count[label_c]+=1
            new_shape = (data_count+1,opt.num_points,opt.feature_dims,)
            dset.resize(new_shape)
            dset_s.resize((data_count+1,opt.num_points,))
            dset[data_count,:,:] = data[c]
            dset_s[data_count,:] = seg_label[c]

            dset.flush()
            dset_s.flush()

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
        return model  

def _snapshot(save_dir, model, epoch, opt):
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
    torch.save(new_state_dict, save_dir+'% of_training_data_at_epoch' + str(epoch) + '.pkl')
    print(f"Save model to {save_dir}_{str(epoch)}.pkl")

def main(opt):
    experiment_id = 'Semantic_segmentation_'+ opt.dataset + '_1024' + '_' + str(opt.percentage)
    snapshot_root = 'snapshot/%s' %experiment_id
    tensorboard_root = 'tensorboard/%s' %experiment_id
    save_dir = os.path.join(ROOT_DIR, snapshot_root, 'models/')
    tboard_dir = os.path.join(ROOT_DIR, tensorboard_root)
    
    #create folder to save trained models
    if opt.model == '':
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            choose = input("Remove " + save_dir + " ? (y/n)")
            if choose == 'y':
                shutil.rmtree(save_dir)
                os.makedirs(save_dir)
            else:
                sys.exit(0)
        if not os.path.exists(tboard_dir):
            os.makedirs(tboard_dir)
        else:
            shutil.rmtree(tboard_dir)
            os.makedirs(tboard_dir)
    sys.stdout = Logger(os.path.join(ROOT_DIR, 'LOG', 'sem_seg_network_log.txt'))
    writer = SummaryWriter(log_dir = tboard_dir)

    #generate part label one-hot correspondence from the catagory:
    if opt.dataset == 's3dis':
        classes = ['ceiling','floor','wall','beam','column','window','door','table','chair','sofa','bookcase','board','clutter']
        class2label = {cls: i for i,cls in enumerate(classes)}
        seg_classes = class2label
        seg_label_to_cat = {}
        for i,cat in enumerate(seg_classes.keys()):
            seg_label_to_cat[i] = cat
    elif opt.dataset == 'arch':
        class2label = {"arch":0, "column":1, "moldings":2, "floor":3, "door_window":4, "wall":5, "stairs":6, "vault":7, "roof":8, "other":9}
        seg_classes = class2label
        seg_label_to_cat = {}
        for i,cat in enumerate(seg_classes.keys()):
            seg_label_to_cat[i] = cat

    # load the dataset
    print('-Preparing dataset...')
    data_resized=False
    train_path = os.path.join(ROOT_DIR, 'cache', 'archfoldnet_1024_%d'%opt.ae_epochs, 'features')
    if(opt.percentage<100):        
        ResizeDataset(path=train_path, percentage=opt.percentage, n_classes=opt.n_classes,shuffle=True)
        data_resized=True
   
    train_dataset = latent_loader.Saved_latent_caps_loader(
            feature_dir=train_path, batch_size=opt.batch_size, npoints=opt.num_points, with_seg=True, shuffle=True, train=True,resized=data_resized)
    test_dataset = latent_loader.Saved_latent_caps_loader(
            feature_dir=train_path, batch_size=opt.batch_size, npoints=opt.num_points, with_seg=True, shuffle=False, train=False,resized=False)


    #initial model
    sem_seg_net = SemSegNet(num_class=opt.n_classes, with_rgb=False)
    #load pretrained model
    if opt.model != '':
        sem_seg_net = load_pretrain(sem_seg_net, opt.model)            
    # load model to gpu
    if opt.gpu_mode:
        sem_seg_net = sem_seg_net.cuda()       
    # initialize optimizer
    optimizer = optim.Adam(sem_seg_net.parameters(), lr=0.01) 

# start training
    n_batch = 0
    # start epoch index
    if opt.model != '':
        start_epoch = op.model[-7:-4]
        if start_epoch[0] == '_':
            start_epoch = start_epoch[1:]
        start_epoch=int(start_epoch)
    else:
        start_epoch = 0
    
    print('training start!!!')
    start_time = time.time()
    loss = []

    for epoch in range(start_epoch, opt.n_epochs):
        batch_id = 0
        sem_seg_net=sem_seg_net.train()
        while train_dataset.has_next_batch():
            latent_caps_, seg_label = train_dataset.next_batch()            
            
            target = torch.from_numpy(seg_label.astype(np.int64))
            
            # concatnate the latent caps with the one hot part label
            latent_caps = torch.from_numpy(latent_caps_).float()
            if(latent_caps.size(0)<opt.batch_size):
                continue
            latent_caps, target = Variable(latent_caps), Variable(target)
            if opt.gpu_mode:
                latent_caps,target = latent_caps.cuda(), target.cuda()                            
    
# forward
            optimizer.zero_grad()
            latent_caps=latent_caps.transpose(2, 1)# consider the capsule vector size as the channel in the network
            output_digit =sem_seg_net(latent_caps)
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
            print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(total_time), epoch, total_time))
            loss.append(train_loss.detach().cpu().numpy())
            writer.add_scalar('Train Loss', loss[-1], epoch)
        
        if epoch % 5 == 0:    
            sem_seg_net=sem_seg_net.eval()    
            correct_sum=0
            batch_id=0
            while test_dataset.has_next_batch():
                latent_caps, seg_label = test_dataset.next_batch()

                target = torch.from_numpy(seg_label.astype(np.int64))
        
                latent_caps = torch.from_numpy(latent_caps).float()
                if(latent_caps.size(0)<opt.batch_size):
                    continue
                latent_caps, target = Variable(latent_caps), Variable(target)    
                if opt.gpu_mode:
                    latent_caps,target = latent_caps.cuda(), target.cuda()
                
                latent_caps=latent_caps.transpose(2, 1)
                output=sem_seg_net(latent_caps)
                output = output.view(-1, opt.n_classes)        
                target= target.view(-1,1)[:,0] 
        
#                print('bactch_no:%d/%d, train_loss: %f ' % (batch_id, len(train_dataloader)/opt.batch_size, train_loss.item()))
               
                pred_choice = output.data.cpu().max(1)[1]
                correct = pred_choice.eq(target.data.cpu()).cpu().sum()                
                correct_sum=correct_sum+correct.item()
                batch_id+=1
            
            _snapshot(save_dir,sem_seg_net, epoch + 1)

            print(' accuracy of epoch %d is: %f' %(epoch,correct_sum/float((batch_id+1)*opt.batch_size * opt.num_points)))
             
        train_dataset.reset()
        test_dataset.reset()
    print("Training finish!... save training results")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=250, help='number of epochs to train for')
    parser.add_argument('--ae_epochs', type=int, default=100, help='choose which pre-trained ae to use')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
    parser.add_argument('--gpu_mode', action='store_true', help='Enables CUDA training')
    parser.add_argument('--feature_dims', type=int, default=1024, help='scale of latent features')
    parser.add_argument('--num_points', type=int, default=2048, help='input point set size')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, default='arch', help='dataset: s3dis, arch')
    parser.add_argument('--percentage', type=int, default=100, help='training cls with percent of training_data')
    parser.add_argument('--n_classes', type=int, default=10, help='semantic classes in all the catagories')
    parser.add_argument('--encoder', type=str, default='foldingnet', help='encoder use')
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--loss', type=str, default='ChamferLoss', choices=['ChamferLoss_m','ChamferLoss'],
                        help='reconstruction loss')

    opt = parser.parse_args()
    print(opt)

    main(opt)