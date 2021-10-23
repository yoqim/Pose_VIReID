import torch
from torch import nn
import torch.nn.functional as F

class Cross_modal_ContrastiveLoss6(nn.Module):
    def __init__(self, margin=0.3):
        super(Cross_modal_ContrastiveLoss6, self).__init__()
        self.margin = margin
    
    def forward(self, modal1_inputs, modal2_inputs, targets):
        centers = []
        centersR = []
        centersT = []

        for i, l in enumerate(targets):
            feat1 = modal1_inputs[targets==l]
            feat2 = modal2_inputs[targets==l]
            
            center_feat = (feat1.mean(dim=0)+feat2.mean(dim=0)) / 2.0

            centers.append(center_feat.unsqueeze(0))
            centersR.append(feat1.mean(dim=0).unsqueeze(0))
            centersT.append(feat2.mean(dim=0).unsqueeze(0))
   
        centers = torch.cat(centers, 0).cuda()
        centersR = torch.cat(centersR, 0).cuda()
        centersT = torch.cat(centersT, 0).cuda()
        
        n = targets.size(0)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        
        dist1 = self.compute_dist(centersR, centers)
        dd1   = torch.sqrt(dist1 + 1e-10)

        dist2 = self.compute_dist(centersT, centers)
        dd2   = torch.sqrt(dist2 + 1e-10)
        
        label = mask.float()
        loss_r = self.compute_loss(dist1, dd1, label)
        loss_t = self.compute_loss(dist2, dd2, label)
        
        return loss_r + loss_t

    def compute_loss(self, d2, d1, label):
        pos = label * torch.pow(d2, 2)
        neg = (1.0-label) * torch.pow(torch.clamp(self.margin-d1, min=0.0), 2)
        loss_contrastive = torch.mean(pos + neg)
        return loss_contrastive

    def compute_dist(self, inputs1, inputs2):
        n = inputs1.size(0)
        dist1 = torch.pow(inputs1, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist2 = torch.pow(inputs2, 2).sum(dim=1, keepdim=True).expand(n, n)
        
        dist = dist1 + dist2.t()
        dist.addmm_(mat1=inputs1, mat2=inputs2.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist


