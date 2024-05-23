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
from .activation_layer import activation_layer

class ClassFeatConv2(nn.Module):
    def __init__(self, sampleNum,down_ratio=1,feat_dim=64):
        super(ClassFeatConv2, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(sampleNum*feat_dim,feat_dim//down_ratio,kernel_size=3,padding=1),
                        nn.BatchNorm2d(feat_dim//down_ratio, momentum=1, affine=True),
                        activation_layer())
        self.layer2 = nn.Sequential(
                        nn.Conv2d(feat_dim//down_ratio,feat_dim,kernel_size=3,padding=1),
                        nn.BatchNorm2d(feat_dim, momentum=1, affine=True),
                        activation_layer())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        return out

