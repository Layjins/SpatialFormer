import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
from torch.autograd import Variable
import numpy as np

from .relation_module import distLinear
from .gcn import SCGCN_self, DGCN, DGCN_self
from .superglue import SelfAttentionalGNN
#from .backbones.rest import Block as RestBlock
from .non_local import NLBlockND, NLBlockSimple
from .neck import ClassSpecific, TaskSpecific


class ProtoInitDistLinear(nn.Module):
    def __init__(self, indim, outdim, init_proto=None):
        super(ProtoInitDistLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias = False)
        #print('init_proto: ', init_proto.size())
        #print('weight: ', self.L.weight.shape)
        # init Linear layer by (mean=0, variance=0.5)
        #self.L.weight.data = torch.from_numpy(np.random.normal(0, 0.5, size=self.L.weight.shape))
        # init Linear layer by proto
        if init_proto != None:
            self.L.weight.data = init_proto
        # weight normalization
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      
        # metric scale
        if outdim <=200:
            self.scale_factor = 2 #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax
        else:
            self.scale_factor = 10 #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist) 
        return scores

class ProtoInitConv1(nn.Module):
    def __init__(self, indim, outdim, init_proto=None):
        super(ProtoInitConv1, self).__init__()
        self.L = nn.Conv2d(indim, outdim, kernel_size=1, bias=False)
        #print('init_proto: ', init_proto.size())
        #print('weight: ', self.L.weight.shape)
        # init Linear layer by proto
        if init_proto != None:
            self.L.weight.data = init_proto.unsqueeze(2).unsqueeze(3)

    def forward(self, x):
        return self.L(x)

class Linear(nn.Module):
    def __init__(self, indim, outdim, init_proto=None):
        super(Linear, self).__init__()
        self.L = nn.Linear(indim, outdim)

    def forward(self, x):
        return self.L(x)

class Conv1(nn.Module):
    def __init__(self, indim, outdim, init_proto=None):
        super(Conv1, self).__init__()
        self.L = nn.Conv2d(indim, outdim, kernel_size=1)

    def forward(self, x):
        return self.L(x)

def proto_constrain_loss(proto, classifier_layer):
    """ 
    Shape: 
    - Input: (class_num, c) 
    - Output: (1)
    """
    # correlation
    class_num, c = proto.size()
    scale_factor = 2
    correlation = scale_factor * classifier_layer(proto) # (class_num, class_num)
    #correlation = scale_factor * torch.matmul(classifier_layer.weight.data, proto.transpose(0, 1))
    # target of redundancy matrix
    redundancy_param = 0.0 # default=0.0
    target_eye = torch.eye(class_num).cuda()
    target = correlation * (1 - target_eye) + target_eye # only consider the self-similarity
    #target = target_eye
    #target[target_eye==0] = redundancy_param # consider the correlation matrix
    # mse loss
    mse_loss = nn.MSELoss()
    loss = mse_loss(correlation.cuda(), target.cuda())
    return loss

class ClasifierFinetune(nn.Module):
    def __init__(self, args, in_feat_c, in_feat_h, in_feat_w, out_dim, in_feat=None, in_feat_onehot_label=None):
        super(ClasifierFinetune, self).__init__()
        """ 
        batch size = 1
        """ 
        self.nKnovel = args.nKnovel
        self.in_feat_c = in_feat_c
        self.in_feat_h = in_feat_h
        self.in_feat_w = in_feat_w
        self.out_dim = out_dim
        self.novel_classifier = args.novel_classifier
        self.novel_classifier_constrain = args.novel_classifier_constrain
        self.novel_feature = args.novel_feature
        self.novel2base_feat = args.novel2base_feat
        # self-attention
        if self.novel_feature == 'DGCN_self':
            self.attention_module = DGCN_self(self.in_feat_c, self.in_feat_c, self.in_feat_h*self.in_feat_w)
        elif self.novel_feature == 'SuperGlue':
            GNN_layers = ['self'] * 1
            #GNN_layers = ['self'] * 3
            self.attention_module = SelfAttentionalGNN(self.in_feat_c, GNN_layers)
        elif self.novel_feature == 'ResT':
            num_heads = 4
            self.attention_module = RestBlock(self.in_feat_c, num_heads=num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                                              drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, apply_transform=False)
        elif self.novel_feature == 'NLBlockND':
            self.attention_module = NLBlockND(in_channels=self.in_feat_c, inter_channels=self.in_feat_c, mode='embedded', dimension=2)
        elif self.novel_feature == 'NLBlockSimple':
            self.attention_module = NLBlockSimple(inplanes=self.in_feat_c, ratio=1.0)
        elif self.novel_feature == 'Task_feat':
            self.attention_module = TaskSpecific(feature_dim=self.in_feat_c, num_queries=self.nKnovel, FFN_method='MLP')
        elif self.novel_feature == 'Coarse_feat':
            self.attention_module = ClassSpecific(feature_dim=self.in_feat_c, num_queries=self.nKnovel, FFN_method='MLP')

        # define classifier head
        if self.novel_classifier == 'Linear':
            #self.clasifier_fine = nn.Linear(self.in_feat_c, self.out_dim)
            self.clasifier_fine = Linear(self.in_feat_c, self.out_dim)
        elif self.novel_classifier == 'Conv1':
            #self.clasifier_fine = nn.Conv2d(self.in_feat_c, self.out_dim, kernel_size=1)
            self.clasifier_fine = Conv1(self.in_feat_c, self.out_dim)
        elif self.novel_classifier == 'distLinear':
            self.clasifier_fine = distLinear(self.in_feat_c, self.out_dim)
        elif self.novel_classifier == 'ProtoInitDistLinear':
            if (in_feat != None) and (in_feat_onehot_label != None):
                # avg pool
                in_feat = in_feat.mean(3)
                in_feat = in_feat.mean(3)
                in_feat = in_feat.unsqueeze(3)
                in_feat = in_feat.unsqueeze(4)
                # get proto
                b, n, c, h, w = in_feat.size()
                in_feat = in_feat.contiguous().view(b, n, -1)
                in_feat_onehot_label = in_feat_onehot_label.transpose(1, 2)
                in_feat = torch.bmm(in_feat_onehot_label, in_feat)
                in_feat = in_feat.div(in_feat_onehot_label.sum(dim=2, keepdim=True).expand_as(in_feat))
                in_feat = in_feat.contiguous().view(b, self.out_dim, c, -1)
                in_feat = in_feat.squeeze(3)
                in_feat = in_feat.squeeze(0)
                in_feat = F.normalize(in_feat, p=2, dim=1, eps=1e-12) # norm2
                # define classifier
                self.init_proto = in_feat
                self.clasifier_fine = ProtoInitDistLinear(self.in_feat_c, self.out_dim, init_proto=in_feat)
            else:
                self.clasifier_fine = ProtoInitDistLinear(self.in_feat_c, self.out_dim)      
        elif self.novel_classifier == 'ProtoInitConv1':
            if (in_feat != None) and (in_feat_onehot_label != None):
                # avg pool
                in_feat = in_feat.mean(3)
                in_feat = in_feat.mean(3)
                in_feat = in_feat.unsqueeze(3)
                in_feat = in_feat.unsqueeze(4)
                # get proto
                b, n, c, h, w = in_feat.size()
                in_feat = in_feat.contiguous().view(b, n, -1)
                in_feat_onehot_label = in_feat_onehot_label.transpose(1, 2)
                in_feat = torch.bmm(in_feat_onehot_label, in_feat)
                in_feat = in_feat.div(in_feat_onehot_label.sum(dim=2, keepdim=True).expand_as(in_feat))
                in_feat = in_feat.contiguous().view(b, self.out_dim, c, -1)
                in_feat = in_feat.squeeze(3)
                in_feat = in_feat.squeeze(0)
                in_feat = F.normalize(in_feat, p=2, dim=1, eps=1e-12) # norm2
                # define classifier
                self.init_proto = in_feat
                self.clasifier_fine = ProtoInitConv1(self.in_feat_c, self.out_dim, init_proto=in_feat)
            else:
                self.clasifier_fine = ProtoInitConv1(self.in_feat_c, self.out_dim)    

    def forward(self, in_feat, ftrain=None, ytest=None):
        b, n, c, h, w = in_feat.size()
        f_novel = in_feat
        if self.novel_feature == 'DGCN_self':
            in_feat = in_feat.view(-1, self.in_feat_c, self.in_feat_h*self.in_feat_w)
            in_feat = self.attention_module(in_feat)
            # norm
            #in_feat = F.normalize(in_feat, p=2, dim=1, eps=1e-12)
            f_novel = in_feat
        elif self.novel_feature == 'SuperGlue':
            f_novel = self.attention_module(in_feat.contiguous().view(b*n, c, h*w))
            f_novel = f_novel.contiguous().view(b, n, c, h, w)
            in_feat = f_novel
        elif self.novel_feature == 'ResT':
            f_novel = self.attention_module(in_feat.contiguous().view(b*n, c, h, w).flatten(2).permute(0, 2, 1), h, w)
            f_novel = f_novel.permute(0, 2, 1).reshape(b, n, c, h, w)
            in_feat = f_novel
        elif self.novel_feature == 'NLBlockND':
            f_novel = self.attention_module(in_feat.contiguous().view(b*n, c, h, w))
            f_novel = f_novel.contiguous().view(b, n, c, h, w)
            in_feat = f_novel
        elif self.novel_feature == 'NLBlockSimple':
            f_novel = self.attention_module(in_feat.contiguous().view(b*n, c, h, w))
            f_novel = f_novel.contiguous().view(b, n, c, h, w)
            in_feat = f_novel
        elif self.novel_feature == 'Task_feat':
            if ftrain != None:
                f_novel = self.attention_module(ftrain, in_feat)
            in_feat = f_novel
        elif self.novel_feature == 'Coarse_feat':
            in_feat = in_feat.contiguous().view(b*n, c, h, w)
            f_novel, _ = self.attention_module(in_feat)
            in_feat = f_novel

        f_novel = f_novel.contiguous().view(b, n, c, h, w)
        in_feat = in_feat.contiguous().view(-1, self.in_feat_c, self.in_feat_h, self.in_feat_w)
        classifier_constrain_loss = None
        if self.novel_classifier == 'Linear' or self.novel_classifier == 'distLinear' or self.novel_classifier == 'ProtoInitDistLinear':
            # avg_pool
            in_feat = F.avg_pool2d(in_feat, in_feat.size()[2:])
            in_feat = in_feat.view(in_feat.size(0), -1)
            finetune_res = self.clasifier_fine(in_feat)
            # classifier constrain
            if self.novel_classifier == 'ProtoInitDistLinear':
                if self.novel_classifier_constrain == 'ProtoMean':
                    classifier_constrain_loss = proto_constrain_loss(self.init_proto, self.clasifier_fine.L)
            if not self.training:
                finetune_res = finetune_res.view(-1, self.out_dim)
                return finetune_res, f_novel, classifier_constrain_loss
        elif self.novel_classifier == 'Conv1' or self.novel_classifier == 'ProtoInitConv1':
            if not self.training:
                #in_feat = F.avg_pool2d(in_feat, in_feat.size()[2:])
                finetune_res = self.clasifier_fine(in_feat)
                finetune_res = finetune_res.view(-1, self.out_dim, self.in_feat_h*self.in_feat_w)
                finetune_res = F.softmax(finetune_res, dim=1)
                similarity_map = finetune_res
                finetune_res = torch.mean(finetune_res, dim=2)
                finetune_res = finetune_res.view(-1, self.out_dim)
                # novel2base_feat
                if self.novel2base_feat == 'ClassWeightAttention':
                    # select max similarity map
                    _, preds = torch.max(finetune_res, 1)
                    # class weight
                    class_weight = self.clasifier_fine.L.weight.data
                    L_norm = torch.norm(class_weight, p=2, dim =1).unsqueeze(1).expand_as(class_weight)
                    class_weight = class_weight.div(L_norm + 0.00001)
                    # select class weight
                    class_weight_attention = class_weight[preds] # (b*n, c)
                    class_weight_attention = class_weight_attention.view(b, n, c).unsqueeze(3).unsqueeze(4) # (b, n, c, 1, 1)
                    # attention
                    f_novel = f_novel * (class_weight_attention + 1)
                elif self.novel2base_feat == 'ClassWeightAttentionGT' and ytest!=None:
                    # select GroundTruth similarity map
                    ytest = ytest.view(-1)
                    # class weight
                    class_weight = self.clasifier_fine.L.weight.data
                    L_norm = torch.norm(class_weight, p=2, dim =1).unsqueeze(1).expand_as(class_weight)
                    class_weight = class_weight.div(L_norm + 0.00001)
                    # select class weight
                    class_weight_attention = class_weight[ytest] # (b*n, c)
                    class_weight_attention = class_weight_attention.view(b, n, c).unsqueeze(3).unsqueeze(4) # (b, n, c, 1, 1)
                    # attention
                    f_novel = f_novel * (class_weight_attention + 1)
                elif self.novel2base_feat == 'ThreshClassAttention':
                    # class weight
                    class_weight = self.clasifier_fine.L.weight.data
                    # all class weight
                    # class_weight = (self.out_dim, c, 1, 1)
                    thresh = 0.5
                    finetune_res_thresh = finetune_res * (finetune_res > thresh)
                    class_weight_attention = torch.matmul(finetune_res_thresh.unsqueeze(1), class_weight.squeeze(2).squeeze(2)).squeeze(1) # (b*n, c)
                    class_weight_attention = class_weight_attention.view(b, n, c).unsqueeze(3).unsqueeze(4) # (b, n, c, 1, 1)
                    # attention
                    attention_norm = torch.norm(class_weight_attention, p=2, dim =2).unsqueeze(2).expand_as(class_weight_attention)
                    class_weight_attention = class_weight_attention.div(attention_norm + 0.00001)
                    f_novel = f_novel * (class_weight_attention + 1)
                elif self.novel2base_feat == 'SimMapAttention':
                    # select max similarity map
                    _, preds = torch.max(finetune_res, 1)
                    map_index = torch.arange(similarity_map.size(0)).cuda()
                    similarity_map = similarity_map.view(-1, self.in_feat_h*self.in_feat_w)
                    similarity_map_preds = similarity_map[map_index * self.out_dim + preds] # (b*n, h*w)
                    #similarity_map_sorts = similarity_map_preds.unsqueeze(1).repeat(1,c,1)
                    #sorts = torch.argsort(similarity_map_sorts, 2)
                    # softmax
                    similarity_map_preds = F.softmax(similarity_map_preds / 0.025, dim=-1)
                    similarity_map_preds = similarity_map_preds.view(b, n, h, w).unsqueeze(2) # (b, n, 1, h, w)
                    # attention
                    f_novel = f_novel * (similarity_map_preds + 1) # mul_add
                    #f_novel = f_novel + similarity_map_preds # add
                    #f_novel = f_novel * similarity_map_preds # mul
                    # sort similarity_map_preds and rerange f_novel
                    #f_novel = f_novel.view(b*n, c, h*w)
                    #f_novel = f_novel.gather(dim=2, index=sorts)
                    #f_novel = f_novel.view(b, n, c, h, w)
                return finetune_res, f_novel, classifier_constrain_loss
            # patch-wise
            finetune_res = self.clasifier_fine(in_feat)
        return finetune_res, f_novel, classifier_constrain_loss

