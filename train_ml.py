import argparse
import sys
import time 
from apex import amp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms

from data_loader import SYSUData4, RegDBData4, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from utils import *
from color import *

from loss.triplet import Cross_modal_ContrastiveLoss6
from models.model_ml3 import embed_net_ml3, DistillKL


parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu',  help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', default='', type=str, 
                    help='resume from checkpoint')
parser.add_argument('--model_path', default='save_model/', type=str, 
                    help='model save path')
parser.add_argument('--log_path', default='log/', type=str, 
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--low-dim', default=6144, type=int,
                    metavar='D', help='feature dimension')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=64, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--drop', default=0.0, type=float,
                    metavar='drop', help='dropout ratio')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--gpu', default='0', type=str,
                      help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--suffix', default='', type=str, help='suffix to add')


args = parser.parse_args() 
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
np.random.seed(0)
cudnn.benchmark = True

dataset = args.dataset
if dataset == 'sysu':
    data_path = 'path to SYSU'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]                          # thermal to visible
elif dataset =='regdb':
    data_path = 'path to RegDB'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]                          # visible to thermal

checkpoint_path = args.model_path + dataset + '/'

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)

suffix = 'ML'
if dataset =='regdb':
    suffix = suffix + '_regdb_trial{}'.format(args.trial)
if len(args.suffix) > 0:
    suffix += args.suffix

test_log_file = open(log_path + suffix + '.txt', "w")
sys.stdout = Logger(log_path  + suffix + '_os.txt')

print('==> Loading data...')
transform_train = transforms.Compose([
    transforms.ToPILImage(),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h,args.img_w)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])

end = time.time()
if dataset =='sysu':
    print_interval = 100
    end_epoch = 150

    trainset = SYSUData4(transform=transform_train)
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
    
    # testing set
    query_img, query_label, query_cam = process_query_sysu(mode = args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(mode = args.mode)
      
elif dataset =='regdb':
    print_interval = 40
    end_epoch = 25

    trainset = RegDBData4(args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
    
    # testing set
    query_img, query_label = process_test_regdb(trial = args.trial, modal = 'visible')
    gall_img, gall_label  = process_test_regdb(trial = args.trial, modal = 'thermal')

gallset = TestData(gall_img, gall_label, transform = transform_test, img_size =(args.img_w,args.img_h))
queryset = TestData(query_img, query_label, transform = transform_test, img_size =(args.img_w,args.img_h))

gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
   
n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('  Dataset {}:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')   
print('Data Loading Time:\t {:.3f}'.format(time.time()-end))


print('==> Building model..')
net = embed_net_ml3(args.low_dim, n_class, drop=args.drop, num_refinement_stages=1)
net.cuda()

best_acc = 0
start_epoch = 0
if len(args.resume)>0:   
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['mAP']
        net.load_state_dict(checkpoint['net'])
        del checkpoint
        print('==> loaded checkpoint {} (epoch {}): mAP: {:.2f}%'.format(args.resume, start_epoch, best_acc*100))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))


tri_wei = 0.1
pose_wei = 7 if dataset == 'regdb' else 3
ts_wei = 1

criterion = nn.CrossEntropyLoss().cuda()
tri_loss = Cross_modal_ContrastiveLoss6().cuda()
mse_loss = nn.MSELoss().cuda()


ignored_params = list(map(id, net.shared.parameters())) \
                + list(map(id, net.feature1.parameters())) \
                + list(map(id, net.feature2.parameters())) \
                + list(map(id, net.feature3.parameters())) \
                + list(map(id, net.feature4.parameters())) \
                + list(map(id, net.feature5.parameters())) \
                + list(map(id, net.feature6.parameters())) \
                + list(map(id, net.feature7.parameters())) \
                + list(map(id, net.feature8.parameters())) \
                + list(map(id, net.feature9.parameters())) \
                + list(map(id, net.feature10.parameters())) \
                + list(map(id, net.feature11.parameters())) \
                + list(map(id, net.feature12.parameters())) \
                + list(map(id, net.classifier1.parameters())) \
                + list(map(id, net.classifier2.parameters())) \
                + list(map(id, net.classifier3.parameters()))\
                + list(map(id, net.classifier4.parameters()))\
                + list(map(id, net.classifier5.parameters()))\
                + list(map(id, net.classifier6.parameters()))  \
                + list(map(id, net.classifier7.parameters())) \
                + list(map(id, net.classifier8.parameters())) \
                + list(map(id, net.classifier9.parameters()))\
                + list(map(id, net.classifier10.parameters()))\
                + list(map(id, net.classifier11.parameters()))\
                + list(map(id, net.classifier12.parameters()))\


base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())
optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1*args.lr},
        {'params': net.shared.parameters(), 'lr': args.lr},
        {'params': net.feature1.parameters(), 'lr': args.lr},
        {'params': net.feature2.parameters(), 'lr': args.lr},
        {'params': net.feature3.parameters(), 'lr': args.lr},
        {'params': net.feature4.parameters(), 'lr': args.lr},
        {'params': net.feature5.parameters(), 'lr': args.lr},
        {'params': net.feature6.parameters(), 'lr': args.lr},
        {'params': net.feature7.parameters(), 'lr': args.lr},
        {'params': net.feature8.parameters(), 'lr': args.lr},
        {'params': net.feature9.parameters(), 'lr': args.lr},
        {'params': net.feature10.parameters(), 'lr': args.lr},
        {'params': net.feature11.parameters(), 'lr': args.lr},
        {'params': net.feature12.parameters(), 'lr': args.lr},
        {'params': net.classifier1.parameters(), 'lr': args.lr},
        {'params': net.classifier2.parameters(), 'lr': args.lr},
        {'params': net.classifier3.parameters(), 'lr': args.lr},
        {'params': net.classifier4.parameters(), 'lr': args.lr},
        {'params': net.classifier5.parameters(), 'lr': args.lr},
        {'params': net.classifier6.parameters(), 'lr': args.lr},
        {'params': net.classifier7.parameters(), 'lr': args.lr},
        {'params': net.classifier8.parameters(), 'lr': args.lr},
        {'params': net.classifier9.parameters(), 'lr': args.lr},
        {'params': net.classifier10.parameters(), 'lr': args.lr},
        {'params': net.classifier11.parameters(), 'lr': args.lr},
        {'params': net.classifier12.parameters(), 'lr': args.lr},
        ],weight_decay=5e-4, momentum=0.9, nesterov=True)


net, optimizer = amp.initialize(net, optimizer, opt_level='O1')

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    # lr_epoch = [5,10,15,25,35,45,55] if dataset == 'regdb' else [30,60,90,120,150,180] 
    lr_epoch = [8,15,25,40,60,80] if dataset == 'regdb' else [12,30,50,75,125,180] 
    if epoch < lr_epoch[0]:
        lr = args.lr
    elif epoch < lr_epoch[1]:
        lr = args.lr * 0.75
    elif epoch < lr_epoch[2]:
        lr = args.lr * 0.5
    elif epoch < lr_epoch[3]:
        lr = args.lr * 0.25
    elif epoch < lr_epoch[4]:
        lr = args.lr * 0.125
    elif epoch < lr_epoch[5]:
        lr = args.lr * 0.1  
    else:
        lr = args.lr * 0.05
    
    print("len of optimizer.param_groups, ",len(optimizer.param_groups))
    for i in range(len(optimizer.param_groups)):
        if i==0:
            optimizer.param_groups[i]['lr'] = 0.1*lr
        else:
            optimizer.param_groups[i]['lr'] = lr
    return lr
     
def train(epoch):
    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    net.train()
    end = time.time()
    for batch_idx, (input1, input2, label1, label2, mask1, mask2) in enumerate(trainloader):
        labels = torch.cat((label1, label2), 0).cuda()
        
        input1 = input1.cuda()
        input2 = input2.cuda()
        label1 = label1.cuda()
        label2 = label2.cuda()

        masks = torch.cat((mask1, mask2), 0).cuda()
        data_time.update(time.time() - end)

        outputs, feat, pmask, teach_fc = net(input1, input2)
        
        loss1 = 0.0
        loss2 = 0.0
        loss_ts = 0.0

        for i in range(len(outputs)):
            b = feat[i].shape[0]

            tmp1 = criterion(outputs[i], labels)
            tmp2 = tri_loss(feat[i][:b//2,:], feat[i][b//2:,:], label1)
            tmp3 = kd_loss(outputs[i], teach_fc)

            loss1 += tmp1
            loss2 += tmp2 * tri_wei
            loss_ts += tmp3 * ts_wei


        h, w = masks.shape[2:]
        pmask = torch.nn.functional.interpolate(pmask, (h,w))
        loss3 = mse_loss(pmask, masks)*pose_wei
        del pmask, masks

        _, predicted = outputs[0].max(1)
        correct += predicted.eq(labels).sum().item()

        loss = loss1 + loss2 + loss3 + loss_ts

        optimizer.zero_grad()  
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), 2*input1.size(0))

        total += labels.size(0)

        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch_idx%print_interval ==0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Accu: {:.2f}' .format(
                  epoch, batch_idx, len(trainloader), 
                  100.*correct/total, batch_time=batch_time, 
                  data_time=data_time))

            print('L_id: {:.3f}, L_tri: {:.3f}, L_mse: {:.3f}, L_ts: {:.3f}'.format(loss1.item(), loss2.item(), loss3.item(), loss_ts.item()))            

def extract_feat(net,data_loader,forward_mode,nsample):
    ptr = 0
    feat = np.zeros((nsample, args.low_dim))
    feat2 = np.zeros((nsample, args.low_dim//2))
    with torch.no_grad():
        for _, (input, _) in enumerate(data_loader):
            batch_num = input.size(0)
            input = input.cuda()

            f_id, f_all = net(input, input, forward_mode)
            feat[ptr:ptr+batch_num,: ] = f_all.detach().cpu().numpy()
            feat2[ptr:ptr+batch_num,: ] = f_id.detach().cpu().numpy()
            ptr += batch_num
    return feat, feat2

def test():   
    net.eval()
    torch.cuda.empty_cache()
    
    gall_feat, gall_feat2 = extract_feat(net,gall_loader,test_mode[0],ngall)
    query_feat, query_feat2 = extract_feat(net,query_loader,test_mode[1],nquery)    

    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    distmat2 = np.matmul(query_feat2, np.transpose(gall_feat2))

    if dataset =='regdb':
        cmc, mAP = eval_regdb(-distmat, query_label, gall_label)
        cmc2, mAP2 = eval_regdb(-distmat2, query_label, gall_label)
    elif dataset =='sysu':
        cmc, mAP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        cmc2, mAP2 = eval_sysu(-distmat2, query_label, gall_label, query_cam, gall_cam)

    return cmc, mAP, cmc2, mAP2
    

print('==> Start Training...')    
for epoch in range(start_epoch, end_epoch):
    print('==> Preparing Data Loader...')

    sampler = IdentitySampler3(trainset.train_color_label, \
        trainset.train_thermal_label, color_pos, thermal_pos, args.batch_size, 4)
    trainset.cIndex = sampler.index1 # color index
    trainset.tIndex = sampler.index2 # thermal index
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size,\
        sampler = sampler, num_workers=args.workers, drop_last =True)

    train(epoch)

    if epoch > 0 and epoch % 10 == 0:
        print ('Test Epoch: {}'.format(epoch), file=test_log_file)
        print (red('Test Epoch: {}'.format(epoch)))

        cmc, mAP, cmc2, mAP2= test()

        sen1 = 'FC:   Rank-1: {:.2%} | Rank-10: {:.2%} | Rank-20: {:.2%}| mAP: {:.2%}'.format(
                cmc[0], cmc[9], cmc[19], mAP)
        sen2 = '[Short] FC:   Rank-1: {:.2%} | Rank-10: {:.2%} | Rank-20: {:.2%}| mAP: {:.2%}'.format(
                cmc2[0], cmc2[9], cmc2[19], mAP2)

        print(red(sen1))
        print(red(sen2))

        print(sen1, file = test_log_file)
        test_log_file.flush()
        
        print(sen2, file = test_log_file)
        test_log_file.flush()
        
        if cmc[0] > best_acc and 'debug' not in suffix:
            best_acc = cmc[0]
            state = {
                'net': net.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')
