import os,cv2
import numpy as np
from torch.utils.data.sampler import Sampler
import sys
import os.path as osp
import scipy.io as scio

def GenIdx(train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k,v in enumerate(train_color_label) if v==unique_label_color[i]]
        color_pos.append(tmp_pos)
        
    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k,v in enumerate(train_thermal_label) if v==unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)
    return color_pos, thermal_pos

    
class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, batchSize):        
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)
        
        sample_color = np.arange(batchSize)
        sample_thermal = np.arange(batchSize)
        N = np.maximum(len(train_color_label), len(train_thermal_label))
        
        for j in range(N//batchSize+1):
            batch_idx = np.random.choice(uni_label, batchSize, replace = False)
            
            for i in range(batchSize):
                sample_color[i]  = np.random.choice(color_pos[batch_idx[i]], 1)
                sample_thermal[i] = np.random.choice(thermal_pos[batch_idx[i]], 1)
            
            if j ==0:
                index1= sample_color
                index2= sample_thermal
            else:
                index1 = np.hstack((index1, sample_color))
                index2 = np.hstack((index2, sample_thermal))
        
        self.index1 = index1
        self.index2 = index2
        self.N  = N
        
    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N          

class IdentitySampler2(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """
    def __init__(self, train_color_label, train_thermal_label, 
                       color_pos, thermal_pos, 
                       batchSize, pics=4):        
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)
        
        sample_color = np.arange(batchSize)
        sample_thermal = np.arange(batchSize)

        self.pics = pics
        batch = batchSize // pics
        
        N = np.maximum(len(train_color_label), len(train_thermal_label))

        for j in range(N//batch+1):
            batch_idx = np.random.choice(uni_label, batch, replace = False)
            
            for i in range(batch):
                temp = i * pics
                sample_color[temp:temp+pics]  = np.random.choice(color_pos[batch_idx[i]], pics)
                sample_thermal[temp:temp+pics] = np.random.choice(thermal_pos[batch_idx[i]], pics)
            if j ==0:
                index1= sample_color
                index2= sample_thermal
            else:
                index1 = np.hstack((index1, sample_color))
                index2 = np.hstack((index2, sample_thermal))
        
        self.index1 = index1
        self.index2 = index2
        self.N  = N
    
    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N * self.pics


class IdentitySampler3(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, batchSize, per_img):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)
        
        sample_color = np.arange(batchSize)
        sample_thermal = np.arange(batchSize)
        N = np.maximum(len(train_color_label), len(train_thermal_label))
        
        #per_img = 4
        per_id = batchSize / per_img
        for j in range(N//batchSize+1):
            batch_idx = np.random.choice(uni_label, int(per_id), replace = False)
            
            for s, i in enumerate(range(0, batchSize, per_img)):
                sample_color[i:i+per_img]  = np.random.choice(color_pos[batch_idx[s]], per_img, replace=False)
                sample_thermal[i:i+per_img] = np.random.choice(thermal_pos[batch_idx[s]], per_img, replace=False)
            
            if j ==0:
                index1= sample_color
                index2= sample_thermal
            else:
                index1 = np.hstack((index1, sample_color))
                index2 = np.hstack((index2, sample_thermal))
        
        self.index1 = index1
        self.index2 = index2
        self.N  = N
        
    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N          

class IdentitySampler5(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, batchSize, per_img):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)
        
        sample_color = np.arange(batchSize)
        sample_thermal = np.arange(batchSize)
        N = np.maximum(len(train_color_label), len(train_thermal_label))
        
        #per_img = 4
        per_id = batchSize / per_img
        for j in range(10):
            batch_idx = uni_label[int(j*per_id) : int((j+1)*per_id)]
            print(batch_idx)
            for s, i in enumerate(range(0, batchSize, per_img)):
                sample_color[i:i+per_img]  = color_pos[batch_idx[s]][0:per_img]
                sample_thermal[i:i+per_img] = thermal_pos[batch_idx[s]][0:per_img]
            
            if j ==0:
                index1= sample_color
                index2= sample_thermal
            else:
                index1 = np.hstack((index1, sample_color))
                index2 = np.hstack((index2, sample_thermal))
        
        self.index1 = index1
        self.index2 = index2
        self.N  = N
        
    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N         


class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
 

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """  
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()
    
    def mkdir_if_missing(self,directory):
        if not osp.exists(directory):
            try:
                os.makedirs(directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise  
            

def draw_rect(img,color_mode):
    img = np.array(img)
    rects = [(0, 0, img.shape[1], img.shape[0])]
    for x, y, w, h in rects:
        cv2.rectangle(img, (x, y), (x+w, y+h), color_mode, 2)
    return img