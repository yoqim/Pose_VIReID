import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torchvision import models
from models.GCM import RefinementStage

############################################################################### 
############################################################################### 
def conv1x1xbnxrelu(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1,stride=stride,padding=0, bias=False),
    nn.BatchNorm2d(out_planes),
    nn.ReLU())

def conv1x1xbn(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1,stride=stride,padding=0, bias=False),
    nn.BatchNorm2d(out_planes))

def conv3x3xbnxrelu(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False),
                         nn.BatchNorm2d(out_planes),
                         nn.ReLU())

class basic_deconv_up(nn.Module):
    def __init__(self, in_channel, out_channel=None, scale = 2):
        super(basic_deconv_up, self).__init__() 
        if out_channel is None:
              out_channel = in_channel
        self.deconv = nn.Sequential(
                        nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, stride = 2,
                                           padding=1,output_padding=1, bias=False),
                        nn.BatchNorm2d(out_channel),
                        nn.ReLU())
    def forward(self,x):
        out = self.deconv(x)
        return out
###############################################################################  
###############################################################################      
def get_resnet_layers1(pretrained = True):
    features = list(models.resnet50(pretrained=pretrained).children())[0:5]
    conv1  = features[0]
    bn  = features[1]
    relu = features[2]
    maxpool = features[3]
    block1 = nn.Sequential(conv1, bn, relu, maxpool)
    block2 = features[4]
    return block1, block2

def get_resnet_layers2(pretrained = True):    
    features = list(models.resnet50(pretrained=pretrained).children())[5:8]

    block1 = features[0]
    block2 = features[1]
    block3 = features[2]
    for mo in block3.modules():
        if isinstance(mo, nn.Conv2d):
            mo.stride = (1, 1)
    return block1, block2, block3
###############################################################################  
###############################################################################  
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
    
    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out * 5
###############################################################################  
###############################################################################  
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        # init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)
###############################################################################  
###############################################################################  

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)
###############################################################################  
###############################################################################  

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class FeatureBlock(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0, relu=True):
        super(FeatureBlock, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)] 
        feat_block += [nn.BatchNorm1d(low_dim)]
        
        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block
    
    def forward(self, x):
        x = self.feat_block(x)
        return x


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=True):
        super(ClassBlock, self).__init__()
        classifier = []       
        if relu:
            classifier += [nn.LeakyReLU(0.1)]
        if dropout:
            classifier += [nn.Dropout(p=dropout)]
        
        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier
    def forward(self, x):
        x = self.classifier(x)
        return x       
###############################################################################  
###############################################################################  
    
def part(x, num_part):
    sx = x.size(2) / num_part
    sx = int(sx)
    kx = x.size(2) - sx * (num_part - 1)
    kx = int(kx)
    x = nn.functional.avg_pool2d(x, kernel_size=(kx, x.size(3)), stride=(sx, x.size(3)))
    x = x.view(x.size(0), x.size(1), x.size(2))
    return x


###############################################################################  
###############################################################################  
class visible_net_resnet(nn.Module):
    def __init__(self):
        super(visible_net_resnet, self).__init__()
        self.b1, self.b2 = get_resnet_layers1(pretrained=True)
    
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        return x

class thermal_net_resnet(nn.Module):
    def __init__(self):
        super(thermal_net_resnet, self).__init__()
        self.b1, self.b2 = get_resnet_layers1(pretrained=True)
    
    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        return x

class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T=1):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, size_average=False) * (self.T**2) / y_s.shape[0]
        return loss

class Shared_net_resnet(nn.Module):
    def __init__(self, num_class, num_refinement_stages=1):
        super(Shared_net_resnet, self).__init__()
        self.b1, self.b2, self.b3 = get_resnet_layers2(pretrained=True)

        self.conv4 = conv3x3xbnxrelu(512, 256)
        self.dconv4 = basic_deconv_up(256, 128)

        self.reduce = conv3x3xbnxrelu(128, 1)

        self.refinement_stages = nn.ModuleList()
        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(128+1, 128, 1, to_onnx=False))
        
        self.conv1 = conv3x3xbnxrelu(128, 256, 2)
        self.conv2 = conv3x3xbnxrelu(256, 512, 2)
        self.conv3 = conv1x1xbn(512, 2048)
        
        self.t_fc = FeatureBlock(4096, 1024, dropout=0.0)
        self.t_cla = ClassBlock(1024, num_class, dropout=0.0)
        self.t_avg = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        b = x.size(0)
        x_s = self.b1(x)                                         # (64,512,36,18) - F_S,m
        
        ######### Pose Branch ###################################
        x_p1 = self.conv4(x_s)                                   # (64,256,36,18) - F_B,m^1
        x_p2 = self.dconv4(x_p1)                                 # (64,128,72,36) - F_B,m^1
        
        stages_output = [self.reduce(x_p2)]
        for refinement_stage in self.refinement_stages:
            dx2, sto = refinement_stage(torch.cat([x_p2, stages_output[-1]], dim=1))
            stages_output.extend(sto)                           # sto: (64,17,72,36)

        tx2 = self.conv1(dx2)                                   # (64,256,36,18) - conv 2 - F_B,m^2
        fusion = self.conv3(self.conv2(tx2+x_p1))
        del dx2,tx2,x_p1,x_p2
        
        ######### ReID branch ###################################
        x_id = self.b2(x_s)                # F_ID,m^1
        x_id = self.b3(x_id)                # F_ID,m^2

        ######### Teacher model ###################################
        t_pool = torch.cat([x_id, fusion], dim=1)
        t_pool = self.t_avg(t_pool)
        t_pool = t_pool.view(t_pool.size(0), t_pool.size(1))
        teach_fc = self.t_fc(t_pool)
        teach_fc = self.t_cla(teach_fc)
        
        ########## Re-ID features
        mask = F.sigmoid(fusion)
        x3 = mask * x_id

        sx26 = part(x3, 6)                  # F_ID,m^3
        sx26 = sx26.chunk(6, 2)
        b = sx26[1].size(0)
        xs1 = sx26[0].contiguous().view(b,-1)
        xs2 = sx26[1].contiguous().view(b,-1)
        xs3 = sx26[2].contiguous().view(b,-1)
        xs4 = sx26[3].contiguous().view(b,-1)
        xs5 = sx26[4].contiguous().view(b,-1)
        xs6 = sx26[5].contiguous().view(b,-1)

        ########## pose features
        fu = part(fusion, 6)               # F_B,m^3
        ssx26 = fu.chunk(6, 2)
        b = ssx26[1].size(0)
        xxs1 = ssx26[0].contiguous().view(b,-1)
        xxs2 = ssx26[1].contiguous().view(b,-1)
        xxs3 = ssx26[2].contiguous().view(b,-1)
        xxs4 = ssx26[3].contiguous().view(b,-1)
        xxs5 = ssx26[4].contiguous().view(b,-1)
        xxs6 = ssx26[5].contiguous().view(b,-1)

        return  xs1, xs2, xs3, xs4, xs5, xs6, xxs1, xxs2, xxs3, xxs4, xxs5, xxs6, stages_output[-1], x3, teach_fc

class embed_net_ml3(nn.Module):
    def __init__(self, low_dim, class_num, drop=0.5, num_refinement_stages=1):
        super(embed_net_ml3, self).__init__()
        self.visible_net = visible_net_resnet()
        self.thermal_net = thermal_net_resnet()
        pool_dim = 2048
        
        low_dim=512
        self.shared = Shared_net_resnet(class_num,num_refinement_stages)
        self.feature1 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature2 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature3 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature4 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature5 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature6 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature7 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature8 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature9 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature10 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature11 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature12 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        
        self.classifier1 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier2 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier3 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier4 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier5 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier6 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier7 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier8 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier9 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier10 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier11 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier12 = ClassBlock(low_dim, class_num, dropout=drop)

        self.vis_feature = FeatureBlock(pool_dim, low_dim)
        self.inf_feature = FeatureBlock(pool_dim, low_dim)

        self.visible_classifier = nn.Linear(low_dim, class_num, bias=False)
        self.infrared_classifier = nn.Linear(low_dim, class_num, bias=False)

        self.visible_classifier_ = nn.Linear(low_dim, class_num, bias=False)
        self.visible_classifier_.weight.requires_grad_(False)
        self.visible_classifier_.weight.data = self.visible_classifier.weight.data

        self.infrared_classifier_ = nn.Linear(low_dim, class_num, bias=False)
        self.infrared_classifier_.weight.requires_grad_(False)
        self.infrared_classifier_.weight.data = self.infrared_classifier.weight.data
        
        self.update_rate = 0.2
        self.update_rate_ = self.update_rate
        
        self.l2norm = Normalize(2)

    def forward(self, x1, x2, modal=0):
        if modal==0:
            xx1 = self.visible_net(x1)
            xx2 = self.thermal_net(x2)
            xx = torch.cat((xx1, xx2),0)
        elif modal==1:
            xx = self.visible_net(x1)
        elif modal==2:
            xx = self.thermal_net(x2)
        
        x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, maps, x_glo, teach_fc = self.shared(xx)
        
        y_0 = self.feature1(x_0)
        y_1 = self.feature2(x_1)
        y_2 = self.feature3(x_2)
        y_3 = self.feature4(x_3)
        y_4 = self.feature5(x_4)
        y_5 = self.feature6(x_5)
        y_6 = self.feature7(x_6)
        y_7 = self.feature8(x_7)
        y_8 = self.feature9(x_8)
        y_9 = self.feature10(x_9)
        y_10 = self.feature11(x_10)
        y_11 = self.feature12(x_11)
        
        out_0 = self.classifier1(y_0)
        out_1 = self.classifier2(y_1)
        out_2 = self.classifier3(y_2)
        out_3 = self.classifier4(y_3)
        out_4 = self.classifier5(y_4)
        out_5 = self.classifier6(y_5)
        out_6 = self.classifier7(y_6)
        out_7 = self.classifier8(y_7)
        out_8 = self.classifier9(y_8)
        out_9 = self.classifier10(y_9)
        out_10 = self.classifier11(y_10)
        out_11 = self.classifier12(y_11)
        
        y_0 = self.l2norm(y_0)
        y_1 = self.l2norm(y_1)
        y_2 = self.l2norm(y_2)
        y_3 = self.l2norm(y_3)
        y_4 = self.l2norm(y_4)
        y_5 = self.l2norm(y_5)
        y_6 = self.l2norm(y_6)
        y_7 = self.l2norm(y_7)
        y_8 = self.l2norm(y_8)
        y_9 = self.l2norm(y_9)
        y_10 = self.l2norm(y_10)
        y_11 = self.l2norm(y_11)
        
        if self.training:
            # x_glo = F.avg_pool2d(x_glo, x_glo.size()[2:])
            # x_glo = x_glo.view(x_glo.size(0), -1)

            # b = x_glo.shape[0]

            # logits_v = self.visible_classifier(self.vis_feature(x_glo[:b//2,...]))
            # logits_i = self.infrared_classifier(self.inf_feature(x_glo[b//2:,...]))
            # logits_m = torch.cat([logits_v, logits_i], 0).float()

            # with torch.no_grad():
            #     self.infrared_classifier_.weight.data = self.infrared_classifier_.weight.data * (1 - self.update_rate) \
            #                                         + self.infrared_classifier.weight.data * self.update_rate
            #     self.visible_classifier_.weight.data = self.visible_classifier_.weight.data * (1 - self.update_rate) \
            #                                         + self.visible_classifier.weight.data * self.update_rate

            #     logits_v_ = self.infrared_classifier_(self.vis_feature(x_glo[:b//2,...]))
            #     logits_i_ = self.visible_classifier_(self.inf_feature(x_glo[b//2:,...]))
            #     logits_m_ = torch.cat([logits_v_, logits_i_], 0).float()
            
            # return (out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11), (y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, y_10, y_11), maps, logits_m, logits_m_, teach_fc
            return (out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11), (y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, y_10, y_11), maps, teach_fc
        else:
            y_ID = torch.cat((y_0, y_1, y_2, y_3, y_4, y_5), 1)
            y_all = torch.cat((y_0, y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8, y_9, y_10, y_11), 1)
            
            return  y_ID, y_all

            # return  y_ID, y_all, (out_0, out_1, out_2, out_3, out_4, out_5)         # only for cam


