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
from .backbones.resnet import resnet12, resnet18, resnet50
from .backbones.wrn import WRN
from .backbones.wrn_mixup_model import wrn28_10
from .backbones.resnext import resnext14_16x4d as ResNeXt
from .backbones.squeezenet import squeezeresnet_v1_1 as SqueezeNet
from .backbones.pyramidnet import PyramidNet, get_pyramidnet
from .backbones.densenet import DenseNet, densenet121
from .backbones.efficientnet import EfficientNet, efficientnet_b0
from .backbones.hrnet import hrnet_w18_small_v1, hrnetv2_w18, hrnetv2_w32


class conv4(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(conv4, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = out.view(out.size(0),-1)
        return out # 64

class conv4_512(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(conv4_512, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,128,kernel_size=3,padding=0),
                        nn.BatchNorm2d(128, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(128,256,kernel_size=3,padding=1),
                        nn.BatchNorm2d(256, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(256,512,kernel_size=3,padding=1),
                        nn.BatchNorm2d(512, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = out.view(out.size(0),-1)
        return out # 512

class conv4_512_s(nn.Module):
    """docstring for ClassName"""
    def __init__(self):
        super(conv4_512_s, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(3,64,kernel_size=3,padding=0,bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.Conv2d(64,64,kernel_size=1,stride=2,bias=False),
                        nn.BatchNorm2d(64))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,128,kernel_size=3,padding=0,bias=False),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(128,128,kernel_size=1,stride=2,bias=False),
                        nn.BatchNorm2d(128))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(128,256,kernel_size=3,padding=1,bias=False),
                        nn.BatchNorm2d(256),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(256,512,kernel_size=3,padding=1,bias=False),
                        nn.BatchNorm2d(512),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = out.view(out.size(0),-1)
        return out # 512

class build_backbone(nn.Module):
    def __init__(self, backbone_name, output_size):
        super(build_backbone, self).__init__()
        self.backbone_name = backbone_name
        self.output_size = output_size
        # backbone
        if self.backbone_name == 'conv4':
            self.backbone = conv4()
            self.out_channel = 64
        elif self.backbone_name == 'resnet12':
            self.backbone = resnet12()
            self.out_channel = 512
        elif self.backbone_name == 'resnet18':
            self.backbone = resnet18()
            self.out_channel = 512
        elif self.backbone_name == 'resnet50':
            self.backbone = resnet50()
            self.out_channel = 2048
        elif self.backbone_name == 'WRN':
            #layers = [3, 4, 6, 3] # 50
            layers = [2, 2, 2, 2] # 18
            #channels_per_layers = [64, 128, 256, 512]
            channels_per_layers = [256, 512, 1024, 2048]
            channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]
            self.backbone = WRN(channels=channels, init_block_channels=64, width_factor=2.0)
            self.out_channel = 2048
        elif self.backbone_name == 'wrn28_10':
            self.backbone = wrn28_10()
            self.out_channel = 2048
        elif self.backbone_name == 'ResNeXt':
            self.backbone = ResNeXt()
            self.out_channel = 2048
        elif self.backbone_name == 'SqueezeNet':
            self.backbone = SqueezeNet()
            self.out_channel = 512
        elif self.backbone_name == 'PyramidNet':
            self.backbone = get_pyramidnet(blocks=12, alpha=360, model_name="pyramidnet12_a360")
            self.out_channel = 424
        elif self.backbone_name == 'densenet121':
            self.backbone = densenet121()
            self.out_channel = 1024
        elif self.backbone_name == 'efficientnet_b0':
            self.backbone = efficientnet_b0()
            self.out_channel = 1280
        elif self.backbone_name == 'hrnet_w18_small_v1':
            self.backbone = hrnet_w18_small_v1()
            self.out_channel = 2048
        elif self.backbone_name == 'hrnetv2_w18':
            self.backbone = hrnetv2_w18()
            self.out_channel = 2048
        elif self.backbone_name == 'hrnetv2_w32':
            self.backbone = hrnetv2_w32()
            self.out_channel = 2048
        # conv1 layer
        self.conv1_layer = nn.Sequential(
                            nn.Conv2d(self.out_channel,self.output_size[0],kernel_size=1, stride=1, padding=0),
                            nn.BatchNorm2d(self.output_size[0]),
                            nn.ReLU())

    def forward(self,x):
        feat = self.backbone(x)
        feat_c, feat_h, feat_w = feat.size(1), feat.size(2), feat.size(3)
        out_c, out_h, out_w = self.output_size[0], self.output_size[1], self.output_size[2]
        # h, w
        #if feat_h*feat_w < out_h*out_w:
        #    feat = F.interpolate(feat, size=(out_h, out_w), mode='bilinear', align_corners=False)
        #elif feat_h*feat_w > out_h*out_w:
        #    feat = nn.AdaptiveMaxPool2d((out_h, out_w))(feat)
        # c
        #if feat_c != out_c:
        #    out = self.conv1_layer(feat)
        #else:
        #    out = feat
        out = feat
        #out = out.view(out.size(0),-1)
        return out

