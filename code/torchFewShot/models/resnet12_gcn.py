import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .activation_layer import activation_layer

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel=3, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.norm_layer = norm_layer
        if kernel == 1:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        elif kernel == 3:
            self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = self.norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = self.norm_layer(planes)
        if kernel == 1:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        elif kernel == 3:
            self.conv3 = conv3x3(planes, planes)
        self.bn3 = self.norm_layer(planes)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = activation_layer()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, kernel=1, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.norm_layer = norm_layer
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = self.norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = self.norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = self.norm_layer(planes * 4)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = activation_layer()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, kernel=3, fpn=False, norm_layer=nn.BatchNorm2d):
        self.inplanes = 64
        self.kernel = kernel
        super(ResNet, self).__init__()
        self.fpn=fpn
        self.norm_layer = norm_layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.norm_layer(64)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = activation_layer()

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1) # stride=1: out(11,11)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # stride=1: out(21,21)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.nFeat = 512 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, self.norm_layer):
                if not isinstance(m, nn.InstanceNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                #nn.AvgPool2d(kernel_size=stride, stride=stride, 
                #    ceil_mode=True, count_include_pad=False),
                #nn.Conv2d(self.inplanes, planes * block.expansion, 
                #    kernel_size=1, stride=1, bias=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.kernel, stride, downsample, norm_layer = self.norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.kernel, norm_layer = self.norm_layer))

        return nn.Sequential(*layers)

    def f_shuffle(self, f):
        B, c, h, w = f.size()
        f = f.contiguous().view(B, c, h*w)
        groups = h
        f = f.view(B,c,groups,h*w//groups).permute(0,1,3,2).contiguous().view(B, c, h, w)
        return f

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.layer1(x)
        #x1 = self.f_shuffle(x1)
        x2 = self.layer2(x1)
        #x2 = self.f_shuffle(x2)
        x3 = self.layer3(x2)
        #x3 = self.f_shuffle(x3)
        x4 = self.layer4(x3)
        #x4 = F.avg_pool2d(x4,kernel_size=2)
        #x4 = F.avg_pool2d(x4, x4.size()[2:])
        if self.fpn:
            #return x1, x2, x3, x4
            return x3, x4
        else:
            return x4

class FPN(nn.Module):
    def __init__(self, out_channel=256, norm_layer=nn.BatchNorm2d):
        super(FPN, self).__init__()
        self.resnet_feature = ResNet(BasicBlock, [1,1,1,1], kernel=3, fpn=True, norm_layer=norm_layer)
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=out_channel, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, out_channel, 1, 1, 0)
        #self.conv3 = nn.Conv2d(128, out_channel, 1, 1, 0)
        #self.conv4 = nn.Conv2d(64, out_channel, 1, 1, 0)
        self.fpn_convs = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
 
    def forward(self, x):
        #layer1, layer2, layer3, layer4 = self.resnet_feature(x)  # channel 64 128 256 512
        layer3, layer4 = self.resnet_feature(x)  # channel 256 512

        P5 = self.conv1(layer4)
        P4_ = self.conv2(layer3)
        #P3_ = self.conv3(layer2)
        #P2_ = self.conv4(layer1)
 
        size4 = P4_.shape[2:]
        #size3 = P3_.shape[2:]
        #size2 = P2_.shape[2:]
 
        P4 = P4_ + F.interpolate(P5, size=size4, mode='nearest')
        #P3 = P3_ + F.interpolate(P4, size=size3, mode='nearest')
        #P2 = P2_ + F.interpolate(P3, size=size2, mode='nearest')
 
        #P5 = self.fpn_convs(P5)
        P4 = self.fpn_convs(P4)
        #P3 = self.fpn_convs(P3)
        #P2 = self.fpn_convs(P2)
 
        #return P2, P3, P4, P5
        return P4, layer4


def resnet12_gcn(norm_layer=nn.BatchNorm2d):
    model = ResNet(BasicBlock, [1,1,1,1], kernel=3, fpn=False, norm_layer=norm_layer) # resnet12
    #model = ResNet(BasicBlock, [2,2,2,2], kernel=3, fpn=False, norm_layer=norm_layer) # resnet18
    return model

def resnet12_gcn_fpn(out_channel=256, norm_layer=nn.BatchNorm2d):
    model = FPN(out_channel=out_channel, norm_layer=norm_layer)
    return model
