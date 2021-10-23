from __future__ import print_function
import argparse
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms

from data_loader import TestData
from data_manager_test import *
from eval_metrics import eval_sysu, eval_regdb
from models.model_ml3 import embed_net_ml3

from utils import *
from color import *
import time 
import scipy.io as scio

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu',  help='dataset name: regdb or sysu]')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--model_path', default='save_model/', type=str, help='model save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--low-dim', default=512, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial')
parser.add_argument('--gpu', default='0', type=str,
                      help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--gall-mode', default='single', type=str, help='single or multi')

args = parser.parse_args() 
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
cudnn.benchmark = True

dataset = args.dataset
if dataset == 'sysu':
    data_path = 'Path to SYSU-MM01'
    test_mode = [1, 2]                          # thermal to visible
    n_class = 395
    n_trial = 3

elif dataset =='regdb':
    data_path = 'Path to RegDB'
    # test_mode = [2, 1]                            # visible -> thermal
    # modal = ['visible', 'thermal']
    test_mode = [1, 2]                          # thermal -> visible
    modal = ['thermal','visible']
    n_class = 206


print('==> Building model..')
net = embed_net_ml3(args.low_dim, n_class, drop=0, num_refinement_stages=1)
net.cuda()

print('==> Resuming from checkpoint..')
checkpoint_path = args.model_path + dataset + '/'
if len(args.resume)>0:   
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
        model_cmc = checkpoint['cmc']
        model_map = checkpoint['mAP']

        print(blue('*'*10+"Model info"+'*'*10))
        print(blue('top-1: {:.2%} | top-10: {:.2%} | top-20: {:.2%}'.format(model_cmc[0], model_cmc[9], model_cmc[19])))
        print(blue('mAP: {:.2%}'.format(model_map)))
        del checkpoint

    else:
        print('==> no checkpoint found at {}'.format(args.resume))
        sys.exit()
net.eval() 


print('==> Loading data..')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()

if dataset =='sysu':
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)

elif dataset =='regdb':
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal=modal[0])
    gall_img, gall_label  = process_test_regdb(data_path, trial=args.trial, modal=modal[1])

    ngall = len(gall_label)

    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w,args.img_h))
    gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
    
nquery = len(query_label)

queryset = TestData(query_img, query_label, transform = transform_test, img_size =(args.img_w, args.img_h))   
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
print('Data Loading Time:\t {:.3f}'.format(time.time()-end))

feature_dim = args.low_dim*12
def extract_feat(net,data_loader,forward_mode,nsample):
    ptr = 0
    feat = np.zeros((nsample, feature_dim))
    feat_id = np.zeros((nsample, feature_dim//2))
    with torch.no_grad():
        for _, (input, _) in enumerate(data_loader):
            batch_num = input.size(0)
            input = input.cuda()
            f_id, f = net(input, input, forward_mode)
            feat[ptr:ptr+batch_num,: ] = f.detach().cpu().numpy()
            feat_id[ptr:ptr+batch_num,: ] = f_id.detach().cpu().numpy()
            ptr += batch_num
    return feat, feat_id

query_feat, query_feat_pool = extract_feat(net,query_loader,test_mode[1],nquery)    

all_cmc = 0
all_mAP = 0 
all_cmc_pool = 0
if dataset =='regdb':
    gall_feat, gall_feat_pool = extract_feat(net,gall_loader,test_mode[0],ngall)
    
    # ALL
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    cmc, mAP  = eval_regdb(-distmat, query_label, gall_label)
    
    # ID
    distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
    cmc_pool, mAP_pool = eval_regdb(-distmat_pool, query_label, gall_label)

    print(' ---- Test Trial: {} ----'.format(args.trial))
    print('ALL : top-1: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
        cmc[0], cmc[9], cmc[19]))
    print('mAP: {:.2%}\n'.format(mAP))

    print('ID : top-1: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(
        cmc_pool[0], cmc_pool[9], cmc_pool[19]))
    print('mAP: {:.2%}\n'.format(mAP_pool))
    
elif dataset =='sysu':
    for trial in range(n_trial):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode = args.mode, gall_mode=args.gall_mode)
        ngall = len(gall_label)
        
        trial_gallset = TestData(gall_img, gall_label, transform = transform_test,img_size =(args.img_w,args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)
        gall_feat, gall_feat_pool = extract_feat(net,trial_gall_loader,test_mode[0],ngall)
        
        distmat = np.matmul(query_feat, np.transpose(gall_feat))
        cmc, mAP  = eval_sysu(-distmat, query_label, gall_label,query_cam, gall_cam)
        
        distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
        cmc_pool, mAP_pool = eval_sysu(-distmat_pool, query_label, gall_label,query_cam, gall_cam)
        if trial ==0:
            all_cmc = cmc
            all_mAP = mAP
            all_cmc_pool = cmc_pool
            all_mAP_pool = mAP_pool
        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_cmc_pool = all_cmc_pool + cmc_pool
            all_mAP_pool = all_mAP_pool + mAP_pool
        
        print(red('*'*10 + 'Test Trial: {}'.format(trial) + '*'*10))
        print('ALL : top-1: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(cmc[0], cmc[9], cmc[19]))
        print('mAP: {:.2%}'.format(mAP))
        print('ID : top-1: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(cmc_pool[0], cmc_pool[9], cmc_pool[19]))
        print('mAP: {:.2%}'.format(mAP_pool))

    cmc = all_cmc/n_trial
    mAP = all_mAP/n_trial

    cmc_pool = all_cmc_pool/n_trial 
    mAP_pool = all_mAP_pool/n_trial
    
    print(red('*'*10 + 'All Average' + '*'*10))
    print('ALL: top-1: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(cmc[0], cmc[9], cmc[19]))
    print('mAP: {:.2%}\n'.format(mAP))
    print('ID: top-1: {:.2%} | top-10: {:.2%}| top-20: {:.2%}'.format(cmc_pool[0], cmc_pool[9], cmc_pool[19]))
    print('mAP: {:.2%}'.format(mAP_pool))
