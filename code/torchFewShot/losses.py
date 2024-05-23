from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self, using_focal_loss=False):
        super(CrossEntropyLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.using_focal_loss = using_focal_loss

    def forward(self, inputs, targets, teacher_preds=None, teacher_targets=None):
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)

        margin = 0 # default = 0
        log_probs = self.logsoftmax(inputs) + margin
        targets = torch.zeros(inputs.size(0), inputs.size(1)).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.unsqueeze(-1)
        targets = targets.cuda()
        if teacher_preds!=None and teacher_targets!=None:
            teacher_preds = teacher_preds.view(teacher_preds.size(0), teacher_preds.size(1), -1)
            teacher_preds = self.softmax(teacher_preds)

            teacher_targets = torch.zeros(teacher_preds.size(0), teacher_preds.size(1)).scatter_(1, teacher_targets.unsqueeze(1).data.cpu(), 1)
            teacher_targets = teacher_targets.unsqueeze(-1)
            teacher_targets = teacher_targets.cuda()
            teacher_targets_org = teacher_targets.repeat(1,1,teacher_preds.size(2))
            teacher_targets = teacher_targets * teacher_preds

            # label mapping
            targets_org = targets.repeat(1,1,inputs.size(2))
            targets = targets * inputs
            targets[targets_org==1] = teacher_targets[teacher_targets_org==1]
            # threshold
            #targets[targets>0.5] = 1
            thres_scale = 100.0
            targets[targets*thres_scale>1] = 1
            targets[targets*thres_scale<1] = 0
        # focal loss
        #using_focal_loss = False
        if self.using_focal_loss:
            gama = 1 # default = 2
            probs = F.softmax(inputs, dim=1)
            loss = (- targets * torch.pow((1-probs), gama) * log_probs).mean(0).sum()
        else:
            loss = (- targets * log_probs).mean(0).sum() 
        return loss / inputs.size(2)

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)

        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(inputs.size(0), inputs.size(1)).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = targets.unsqueeze(-1)
        targets = targets.cuda()
        loss = (torch.pow(targets-log_probs, 2)).mean(0).sum()
        return loss / inputs.size(2)

class DistillKL(nn.Module):
    """KL divergence for distillation"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t, y_gt=None):
        # KL mask
        if y_gt != None: 
            y_gt = y_gt.contiguous().view(-1)
            if y_t.dim() == 4:
                b,n,h,w= y_t.size()
                y_t_pred = y_t.contiguous().view(b, n, h*w).mean(2)
            else:
                y_t_pred = y_t
            b,n= y_t_pred.size()
            _, y_t_pred = torch.max(y_t_pred, 1)
            KL_mask = torch.zeros(b)
            KL_mask[y_t_pred == y_gt] = 1
            KL_mask = KL_mask.cuda()
            KL_mask = KL_mask.unsqueeze(-1)
            KL_mask = KL_mask.repeat(1,n) # (b,n)
        # same backbone
        if y_s.dim() == 4:
            b,n,h,w= y_s.size()
            if y_gt != None: 
                KL_mask = KL_mask.unsqueeze(-1)
                KL_mask = KL_mask.repeat(1,1,h*w) # (b,n,h*w)
                y_s = KL_mask * y_s.contiguous().view(b, n, h*w)
                y_t = KL_mask * y_t.contiguous().view(b, n, h*w)
            y_s = y_s.contiguous().view(b, n, h*w).transpose(1,2).contiguous().view(b*h*w, n)
            y_t = y_t.contiguous().view(b, n, h*w).transpose(1,2).contiguous().view(b*h*w, n)
        else:
            if y_gt != None: 
                y_s = KL_mask * y_s
                y_t = KL_mask * y_t

        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss

class FeatMixLoss(nn.Module):
    def __init__(self, using_focal_loss=False):
        super(FeatMixLoss, self).__init__()
        self.CrossEntropyLoss = CrossEntropyLoss(using_focal_loss=using_focal_loss)

    def forward(self, pred, pids_a, pids_b, mix_ratio):
        loss_res = mix_ratio * self.CrossEntropyLoss(pred, pids_a) + (1 - mix_ratio) * self.CrossEntropyLoss(pred, pids_b)
        return loss_res

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2, min_weight=0):
        super(AutomaticWeightedLoss, self).__init__()
        self.num = num
        self.min_weight = min_weight
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        weights = []
        bias = []
        for i, loss in enumerate(x):
            weights_i = 0.5 / (self.params[i] ** 2) + self.min_weight
            bias_i = torch.log(1 + self.params[i] ** 2)
            loss_sum += weights_i * loss + bias_i
            weights.append(weights_i)
            bias.append(bias_i)
        return loss_sum, weights, bias

class AutomaticMetricLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2, init_weight=1.0, min_weights=[0,0]):
        super(AutomaticMetricLoss, self).__init__()
        self.num = num
        self.min_weights = min_weights
        params = torch.ones(num, requires_grad=True) * init_weight
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        weights = []
        bias = []
        for i, loss in enumerate(x):
            weights_i = 0.5 / (self.params[i] ** 2) + self.min_weights[i]
            bias_i = torch.log(1 + self.params[i] ** 2)
            loss_sum += weights_i * loss + bias_i
            weights.append(weights_i)
            bias.append(bias_i)
        return loss_sum, weights, bias
