from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CrossEntropyLoss(nn.Module):
    def __init__(self, using_focal_loss=False):
        super(CrossEntropyLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.using_focal_loss = using_focal_loss

    def forward(self, inputs, targets):
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)

        margin = 0 # default = 0
        log_probs = self.logsoftmax(inputs) + margin
        targets = torch.zeros(inputs.size(0), inputs.size(1)).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.unsqueeze(-1)
        targets = targets.cuda()
        # focal loss
        #using_focal_loss = False
        if self.using_focal_loss:
            gama = 1 # default = 2
            probs = F.softmax(inputs, dim=1)
            loss = (- targets * torch.pow((1-probs), gama) * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).mean(0).sum() 
        return loss / inputs.size(2)

class GlobalFeatMixLoss(nn.Module):
    def __init__(self, backbone_outsize, num_classes, mix_alpha=2.0, using_focal_loss=False):
        super(GlobalFeatMixLoss, self).__init__()
        self.backbone_outsize = backbone_outsize
        self.num_classes = num_classes
        self.mix_alpha = mix_alpha
        # classifier
        self.clasifier = nn.Conv2d(self.backbone_outsize[0], self.num_classes, kernel_size=1)
        # loss
        self.CrossEntropyLoss = CrossEntropyLoss(using_focal_loss=using_focal_loss)

    def forward(self, in_features, pids):
        B, c, h, w = in_features.size() # B=b*n2
        pids = pids.view(-1)
        index=np.arange(B)
        np.random.shuffle(index)
        in_features_rand = in_features[index,:,:,:]
        pids_rand = pids[index]
        y_a = pids
        y_b = pids_rand
        # feature mix
        alpha = self.mix_alpha
        lam = np.random.beta(alpha, alpha)
        mix_ratio = lam
        mix_features = mix_ratio * in_features + (1 - mix_ratio) * in_features_rand
        pred = self.clasifier(mix_features)
        # mix loss
        loss_res = mix_ratio * self.CrossEntropyLoss(pred, y_a) + (1 - mix_ratio) * self.CrossEntropyLoss(pred, y_b)
        return loss_res

