
#
#
#      0=================================0
#      |    Project Name                 |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Implements: Support Vector Machine
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO - 2020/12/26 16:31 PM
#
#


# ----------------------------------------
# Import packages and constant
# ----------------------------------------
import os
import sys
import h5py
import numpy as np
from glob import glob
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns; sns.set()
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'datasets'))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

def ResizeDataset(path, percentage, n_classes, shuffle):
    if path == 'cache/modelnet40/features/':
        original_name = ['train0.h5', 'train1.h5', 'train2.h5', 
        'train3.h5', 'train4.h5']
    elif path == 'cache/shapenetpart/features/':
        original_name = ['train0.h5', 'train1.h5', 'train2.h5', 
        'train3.h5', 'train4.h5', 'train5.h5', 'train6.h5', 'train7.h5']
    elif path == 'cache/shapenetcore/features/':
        original_name = ['train0.h5', 'train1.h5', 'train2.h5', 
        'train3.h5', 'train4.h5', 'train5.h5', 'train6.h5', 'train7.h5']
    else:
        original_name = ['train0.h5', 'train1.h5', 'train2.h5', 
        'train3.h5', 'train4.h5', 'train5.h5', 'train6.h5', 'train7.h5']
    
    for h5_name in original_name:
        ori_name = os.path.join(path, h5_name)
        out_file_name= ori_name + "_" + str(percentage)+ "_resized.h5"

        if os.path.exists(out_file_name):
            os.remove(out_file_name)
        fw = h5py.File(out_file_name, 'w', libver='latest')
        dset = fw.create_dataset("data", (1,1024,),maxshape=(None,1024), dtype='<f4')
        dset_l = fw.create_dataset("label",(1,),maxshape=(None,),dtype='uint8')
        fw.swmr_mode = True   
        f = h5py.File(ori_name)
        data = f['data'][:]
        cls_label = f['label'][:]
    
        #data shuffle
        if shuffle:        
            idx = np.arange(len(cls_label))
            np.random.shuffle(idx)
            data,cls_label = data[idx, ...], cls_label[idx]
    
        class_dist= np.zeros(n_classes)
        for c in range(len(data)):
            class_dist[cls_label[c]]+=1
        log_string('Ori data to size of :', np.sum(class_dist))
        log_string ('class distribution of this dataset :',class_dist)
        
        class_dist_new= (percentage*class_dist/100).astype(int)
        for i in range(n_classes):
            if class_dist_new[i]<1:
                class_dist_new[i]=1
        class_dist_count=np.zeros(n_classes)

        data_count=0
        for c in range(len(data)):
            label_c=cls_label[c]
            if(class_dist_count[label_c] < class_dist_new[label_c]):
                class_dist_count[label_c]+=1
                new_shape = (data_count+1,1024,)
                dset.resize(new_shape)
                dset_l.resize((data_count+1,))
                dset[data_count,:] = data[c]
                dset_l[data_count] = cls_label[c]
                dset.flush()
                dset_l.flush()
                data_count+=1
        log_string('Finished resizing data to size of :', np.sum(class_dist_new))
        log_string ('class distribution of resized dataset :',class_dist_new)
        fw.close


# Read in the list of categories in MODELNET40
def get_category_names(dataset_name):
    if dataset_name == 'modelnet40':
        shape_names_file = os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'shape_names.txt')
    elif dataset_name == 'shapenetcore':
        shape_names_file = os.path.join(DATA_DIR, 'shapenetcorev2_hdf5_2048', 'shape_names.txt')
    else:
        shape_names_file = os.path.join(DATA_DIR, 'arc', 'shape_names.txt')
    shape_names = [line.rstrip() for line in open(shape_names_file)]
    return shape_names

    
class SVM(object):
    def __init__(self, feature_dir, percent, dataset_name):
        self.feature_dir = feature_dir

        if(percent<100):
            ResizeDataset(path = self.feature_dir, percentage=percent, n_classes=40, shuffle=True)
            self.train_path = glob(os.path.join(self.feature_dir, 'train*%s_resized.h5'%percent))
        else:
            self.train_path = glob(os.path.join(self.feature_dir, 'train*.h5'))

        self.test_path = glob(os.path.join(self.feature_dir, 'test*.h5'))

        print("Loading feature dataset...")
        train_data = []
        train_label = []
        for path in self.train_path:
            f = h5py.File(path, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            train_data.append(data)
            train_label.append(label)
        self.train_data = np.concatenate(train_data, axis=0)
        self.train_label = np.concatenate(train_label, axis=0)
        print("Training set size:", np.size(self.train_data, 0))

        test_data = []
        test_label = []
        for path in self.test_path:
            f = h5py.File(path, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            test_data.append(data)
            test_label.append(label)
        self.test_data = np.concatenate(test_data, axis=0)
        self.test_label = np.concatenate(test_label, axis=0)
        print("Testing set size:", np.size(self.test_data, 0))

    def classify(self):
        clf = LinearSVC(random_state=0)
        clf.fit(self.train_data, self.train_label)
        result = clf.predict(self.test_data)
        accuracy = np.sum(result==self.test_label).astype(float) / np.size(self.test_label)
        log_string(str(classification_report(self.test_label, result,
                            target_names=get_category_names())))
        mat = confusion_matrix(self.test_label, result)
        plt.figure(figsize=(10, 16))
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, cmap='YlOrRd',
            xticklabels=get_category_names(dataset_name),
            yticklabels=get_category_names(dataset_name))
        plt.xlabel('true label')
        plt.ylabel('predicted label')
        plt.savefig("output/heatmap_%s.png"%percent, dpi=300)

        log_string("Transfer linear SVM accuracy: {:.2f}%".format(accuracy*100))

LOG_FOUT = open(os.path.join(ROOT_DIR, 'LOG','svm_log.txt'), 'w')
def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)
