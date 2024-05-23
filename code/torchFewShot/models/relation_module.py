import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
import os
import math
import argparse
import scipy as sp
import scipy.stats
from .blocks import *
from torch.nn.utils.weight_norm import WeightNorm
from .activation_layer import activation_layer


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

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

class EuclideanDistLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(EuclideanDistLinear, self).__init__()
        self.L = nn.Linear( indim, outdim, bias = False)
        self.class_wise_learnable_norm = True  #See the issue#4&8 in the github 
        if self.class_wise_learnable_norm:      
            WeightNorm.apply(self.L, 'weight', dim=0) #split the weight update component to direction and norm      

        if outdim <=200:
            self.scale_factor = 2; #a fixed scale factor to scale the output of cos value into a reasonably large input for softmax
        else:
            self.scale_factor = 10; #in omniglot, a larger scale factor is required to handle >1000 output classes.

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim =1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm+ 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(self.L.weight.data, p=2, dim =1).unsqueeze(1).expand_as(self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        cos_dist = self.L(x_normalized) #matrix product by forward function, but when using WeightNorm, this also multiply the cosine distance by a class-wise learnable norm, see the issue#4&8 in the github
        scores = self.scale_factor* (cos_dist) 

        return scores

class PrototypeNet(nn.Module):
    def __init__(self, metric_scale=False):
        super(PrototypeNet, self).__init__()
        # metric scaling
        self.metric_scale = metric_scale
        if self.metric_scale:
            self.metric_scale_param = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        
    def forward(self, sample_features_ext, batch_features_ext):     
        dists = torch.pow(sample_features_ext-batch_features_ext, 2)
        dists = dists.view(batch_features_ext.size(0),batch_features_ext.size(1),-1)
        dists = torch.mean(dists, 2)
        # metric scaling
        if self.metric_scale:
            dists = dists * self.metric_scale_param
        
        return -dists
    
class CosinNet(nn.Module):
    def __init__(self,  metric_scale=False):
        super(CosinNet, self).__init__()
        # metric scaling
        self.metric_scale = metric_scale
        if self.metric_scale:
            self.metric_scale_param = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        
    def forward(self, sample_features_ext, batch_features_ext):     
        sample_features_ext = sample_features_ext.view(batch_features_ext.size(0),batch_features_ext.size(1),-1)
        batch_features_ext = batch_features_ext.view(batch_features_ext.size(0),batch_features_ext.size(1),-1)
        cosin = torch.cosine_similarity(sample_features_ext, batch_features_ext, dim=2)
        # metric scaling
        if self.metric_scale:
            cosin = cosin * self.metric_scale_param
        
        return cosin

class RelationNet(nn.Module):
    """docstring for RelationNet"""
    def __init__(self,input_size=[64,19,19],middle_dim=64,hidden_size=8, metric_scale=False):
        super(RelationNet, self).__init__()
        #input_size=[c, w, h] 
        padding = 1 if ( input_size[1] <10 ) and ( input_size[2] <10 ) else 0 # when using Resnet, conv map without avgpooling is 7x7, need padding in block to do pooling
        shrink_s = lambda s: int((int((s- 2 + 2*padding)/2)-2 + 2*padding)/2)

        self.relu = activation_layer()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(input_size[0]*2,middle_dim,kernel_size=3,padding=padding),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(middle_dim,middle_dim,kernel_size=3,padding=padding),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer(),
                        nn.MaxPool2d(2))
        #self.layer3 = nn.Conv2d(middle_dim, 1, kernel_size=1)
        self.fc1 = nn.Linear(middle_dim* shrink_s(input_size[1]) * shrink_s(input_size[2]),hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

        # metric scaling
        self.metric_scale = metric_scale
        if self.metric_scale:
            self.metric_scale_param = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

    def forward(self,x):
        #out = nn.AdaptiveAvgPool2d((1, 1))(x)
        out = self.layer1(x)
        out = self.layer2(out)
        # fc clasifier
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        # conv1 clasifier
        #out = self.layer3(out)
        # metric scaling
        if self.metric_scale:
            out = out * self.metric_scale_param
        return out

class RelationNetPatch(nn.Module):
    """docstring for RelationNet"""
    def __init__(self,input_size=[64,19,19],middle_dim=64,hidden_size=8, metric_scale=False):
        super(RelationNetPatch, self).__init__()
        self.relu = activation_layer()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(input_size[0]*2,middle_dim,kernel_size=1),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer())
        self.layer2 = nn.Sequential(
                        nn.Conv2d(middle_dim,middle_dim,kernel_size=1),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer())
        self.layer3 = nn.Conv2d(middle_dim, 1, kernel_size=1)

    def forward(self,x):
        #out = nn.AdaptiveAvgPool2d((1, 1))(x)
        out = self.layer1(x)
        out = self.layer2(out)
        # conv1 clasifier
        out = self.layer3(out)
        return out

class RelationNetPatch_v1(nn.Module):
    """docstring for RelationNet"""
    def __init__(self,input_size=[64,19,19],middle_dim=64,hidden_size=8, metric_scale=False):
        super(RelationNetPatch_v1, self).__init__()
        self.relu = activation_layer()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(input_size[0]*2,middle_dim,kernel_size=3,padding=1),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer())
        self.layer2 = nn.Sequential(
                        nn.Conv2d(middle_dim,middle_dim,kernel_size=3,padding=1),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer())
        self.layer3 = nn.Conv2d(middle_dim, 1, kernel_size=1)

    def forward(self,x):
        #out = nn.AdaptiveAvgPool2d((1, 1))(x)
        out = self.layer1(x)
        out = self.layer2(out)
        # conv1 clasifier
        out = self.layer3(out)
        return out

class RelationNetPatch_v2(nn.Module):
    """docstring for RelationNet"""
    def __init__(self,input_size=[64,19,19],middle_dim=64,hidden_size=8, metric_scale=False):
        super(RelationNetPatch_v2, self).__init__()
        self.relu = activation_layer()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(input_size[0]*2,middle_dim,kernel_size=1),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer())
        self.layer2 = nn.Sequential(
                        nn.Conv2d(middle_dim,middle_dim,kernel_size=1),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer())
        self.layer3 = nn.Conv2d(middle_dim, hidden_size, kernel_size=1)
        self.layer4 = nn.Conv2d(hidden_size, 1, kernel_size=1)

    def forward(self,x):
        #out = nn.AdaptiveAvgPool2d((1, 1))(x)
        out = self.layer1(x)
        out = self.layer2(out)
        # conv1 clasifier
        out = F.relu(self.layer3(out))
        out = self.layer4(out)
        return out

class RelationNetPatch_v3(nn.Module):
    """docstring for RelationNet"""
    def __init__(self,input_size=[64,19,19],middle_dim=64,hidden_size=8, metric_scale=False):
        super(RelationNetPatch_v3, self).__init__()
        self.relu = activation_layer()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(input_size[0]*2,middle_dim,kernel_size=3,padding=1),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer())
        self.layer2 = nn.Sequential(
                        nn.Conv2d(middle_dim,middle_dim,kernel_size=3,padding=1),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer())
        self.layer3 = nn.Conv2d(middle_dim, hidden_size, kernel_size=1)
        self.layer4 = nn.Conv2d(hidden_size, 1, kernel_size=1)

    def forward(self,x):
        #out = nn.AdaptiveAvgPool2d((1, 1))(x)
        out = self.layer1(x)
        out = self.layer2(out)
        # conv1 clasifier
        out = F.relu(self.layer3(out))
        out = self.layer4(out)
        return out

class RelationNet234(nn.Module):
    """docstring for RelationNet234"""
    def __init__(self,input_size=[64,19,19],middle_dim=64,hidden_size=8, metric_scale=False):
        super(RelationNet234, self).__init__()
        #input_size=[c, w, h] 
        padding = 1 if ( input_size[1] <10 ) and ( input_size[2] <10 ) else 0 # when using Resnet, conv map without avgpooling is 7x7, need padding in block to do pooling
        shrink_s = lambda s: int((int((s- 2 + 2*padding)/2)-2 + 2*padding)/2)

        self.relu = activation_layer()

        #metric1
        self.layer1_1 = nn.Sequential(
                        nn.Conv2d(input_size[0]*2,middle_dim,kernel_size=3,padding=padding),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer(),
                        nn.MaxPool2d(2))
        self.layer1_2 = nn.Sequential(
                        nn.Conv2d(middle_dim,middle_dim,kernel_size=3,padding=padding),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer(),
                        nn.MaxPool2d(2))
        # metric2
        self.layer2_1 = nn.Sequential(
                        nn.Conv2d(input_size[0]*2,middle_dim,kernel_size=3,padding=padding),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer(),
                        nn.MaxPool2d(2))
        self.layer2_2 = nn.Sequential(
                        nn.Conv2d(middle_dim,middle_dim,kernel_size=3,padding=1),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer())
        self.layer2_3 = nn.Sequential(
                        nn.Conv2d(middle_dim,middle_dim,kernel_size=3,padding=padding),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer(),
                        nn.MaxPool2d(2))
        # metric3
        self.layer3_1 = nn.Sequential(
                        nn.Conv2d(input_size[0]*2,middle_dim,kernel_size=3,padding=padding),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer(),
                        nn.MaxPool2d(2))
        self.layer3_2 = nn.Sequential(
                        nn.Conv2d(middle_dim,middle_dim,kernel_size=3,padding=1),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer())
        self.layer3_3 = nn.Sequential(
                        nn.Conv2d(middle_dim,middle_dim,kernel_size=3,padding=1),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer())
        self.layer3_4 = nn.Sequential(
                        nn.Conv2d(middle_dim,middle_dim,kernel_size=3,padding=padding),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer(),
                        nn.MaxPool2d(2))
        
        #self.fc1_share = nn.Linear(middle_dim*3*3,hidden_size)
        self.fc1 = nn.Linear(middle_dim* shrink_s(input_size[1]) * shrink_s(input_size[2]),hidden_size)
        self.fc2 = nn.Linear(middle_dim* shrink_s(input_size[1]) * shrink_s(input_size[2]),hidden_size)
        self.fc3 = nn.Linear(middle_dim* shrink_s(input_size[1]) * shrink_s(input_size[2]),hidden_size)
        self.fc_share = nn.Linear(hidden_size*3,1)

        # metric scaling
        self.metric_scale = metric_scale
        if self.metric_scale:
            self.metric_scale_param = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

    def forward(self,x):
        #metric1
        out1 = self.layer1_1(x)
        out1 = self.layer1_2(out1)
        out1 = out1.view(out1.size(0),-1)
        out1 = F.relu(self.fc1(out1))
        #metric2
        out2 = self.layer2_1(x)
        out2 = self.layer2_2(out2)
        out2 = self.layer2_3(out2)
        out2 = out2.view(out2.size(0),-1)
        out2 = F.relu(self.fc2(out2))
        #metric3
        out3 = self.layer3_1(x)
        out3 = self.layer3_2(out3)
        out3 = self.layer3_3(out3)
        out3 = self.layer3_4(out3)
        out3 = out3.view(out3.size(0),-1)
        out3 = F.relu(self.fc3(out3))
        #concat
        out = torch.cat((out1, out2, out3), 1)
        out = self.fc_share(out)
        # metric scaling
        #if self.metric_scale:
        #    out = out * self.metric_scale_param
        return out

class RelationNet234Patch(nn.Module):
    """docstring for RelationNet234"""
    def __init__(self,input_size=[64,19,19],middle_dim=64,hidden_size=8, metric_scale=False):
        super(RelationNet234Patch, self).__init__()
        self.relu = activation_layer()

        #metric1
        self.layer1_1 = nn.Sequential(
                        nn.Conv2d(input_size[0]*2,middle_dim,kernel_size=1),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer())
        self.layer1_2 = nn.Sequential(
                        nn.Conv2d(middle_dim,middle_dim,kernel_size=1),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer())
        # metric2
        self.layer2_1 = nn.Sequential(
                        nn.Conv2d(input_size[0]*2,middle_dim,kernel_size=1),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer())
        self.layer2_2 = nn.Sequential(
                        nn.Conv2d(middle_dim,middle_dim,kernel_size=1),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer())
        self.layer2_3 = nn.Sequential(
                        nn.Conv2d(middle_dim,middle_dim,kernel_size=1),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer())
        # metric3
        self.layer3_1 = nn.Sequential(
                        nn.Conv2d(input_size[0]*2,middle_dim,kernel_size=1),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer())
        self.layer3_2 = nn.Sequential(
                        nn.Conv2d(middle_dim,middle_dim,kernel_size=1),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer())
        self.layer3_3 = nn.Sequential(
                        nn.Conv2d(middle_dim,middle_dim,kernel_size=1),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer())
        self.layer3_4 = nn.Sequential(
                        nn.Conv2d(middle_dim,middle_dim,kernel_size=1),
                        nn.BatchNorm2d(middle_dim, momentum=1, affine=True),
                        activation_layer())
        self.fc1 = nn.Linear(middle_dim,hidden_size)
        self.fc2 = nn.Linear(middle_dim,hidden_size)
        self.fc3 = nn.Linear(middle_dim,hidden_size)
        self.fc_share = nn.Linear(hidden_size*3,1)

    def forward(self,x):
        #metric1
        out1 = self.layer1_1(x)
        out1 = self.layer1_2(out1)
        out1 = out1.view(out1.size(0),-1)
        out1 = F.relu(self.fc1(out1))
        #metric2
        out2 = self.layer2_1(x)
        out2 = self.layer2_2(out2)
        out2 = self.layer2_3(out2)
        out2 = out2.view(out2.size(0),-1)
        out2 = F.relu(self.fc2(out2))
        #metric3
        out3 = self.layer3_1(x)
        out3 = self.layer3_2(out3)
        out3 = self.layer3_3(out3)
        out3 = self.layer3_4(out3)
        out3 = out3.view(out3.size(0),-1)
        out3 = F.relu(self.fc3(out3))
        #concat
        out = torch.cat((out1, out2, out3), 1)
        out = self.fc_share(out)
        return out

class ClassifierNet(nn.Module):
    """docstring for ClassifierNet"""
    def __init__(self,input_size, class_num, metric_scale=False):
        super(ClassifierNet, self).__init__()
        #self.cbam = CBAM(64)
        self.dist1 = distLinear(input_size, class_num)
        # metric scaling
        self.metric_scale = metric_scale
        if self.metric_scale:
            self.metric_scale_param = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

    def forward(self,x):
        #x = self.cbam(x)
        out = x.view(x.size(0),-1)
        out = self.dist1(out)
        # metric scaling
        #if self.metric_scale:
        #    out = out * self.metric_scale_param
        return out

class GlobalNet(nn.Module):
    """from CAN"""
    def __init__(self,input_size, class_num):
        super(GlobalNet, self).__init__()
        #self.clasifier = nn.Conv2d(input_size, class_num, kernel_size=1) 
        self.clasifier = nn.Linear(input_size, class_num)

    def forward(self,x):
        #out = x.view(x.size(0),-1,1,1)
        out = x.view(x.size(0),-1)
        out = self.clasifier(out)
        #out = out.view(out.size(0),-1)
        return out

