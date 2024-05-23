from __future__ import absolute_import
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import functional as F
#from .activation_layer import activation_layer


class ConvBlock(nn.Module):
    """Basic convolutional block:
    convolution + batch normalization.

    Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
    - in_c (int): number of input channels.
    - out_c (int): number of output channels.
    - k (int or tuple): kernel size.
    - s (int or tuple): stride.
    - p (int or tuple): padding.
    """
    def __init__(self, in_c, out_c, k, s=1, p=0, norm_layer=nn.BatchNorm2d):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
        self.bn = norm_layer(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class CAM(nn.Module):
    def __init__(self, backbone_outsize=[512,6,6], norm_layer=nn.BatchNorm2d):
        super(CAM, self).__init__()
        self.norm_layer = norm_layer
        self.conv1 = ConvBlock(backbone_outsize[1]*backbone_outsize[2], backbone_outsize[1], 1, norm_layer=self.norm_layer)
        self.conv2 = nn.Conv2d(backbone_outsize[1], backbone_outsize[1]*backbone_outsize[2], 1, stride=1, padding=0)
        #self.relu = activation_layer()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

    def get_attention(self, a):
        input_a = a # (b, n1, n2, h*w, h*w)

        a = a.mean(3) # (b, n1, n2, h*w)
        a = a.transpose(1, 3) # (b, h*w, n2, n1)
        a = F.relu(self.conv1(a)) # (b, h, n2, n1)
        #a = self.relu(self.conv1(a)) 
        a = self.conv2(a) # (b, h*w, n2, n1)
        a = a.transpose(1, 3)  # (b, n1, n2, h*w)
        a = a.unsqueeze(3)  # (b, n1, n2, 1, h*w)
        a = torch.mean(input_a * a, -1) # (b, n1, n2, h*w)
        a = F.softmax(a / 0.025, dim=-1) + 1
        return a 

    def forward(self, f1, f2):
        b, n1, c, h, w = f1.size()
        n2 = f2.size(1)

        f1 = f1.view(b, n1, c, -1) # (b, n1, c, h*w)
        f2 = f2.view(b, n2, c, -1) # (b, n2, c, h*w)

        f1_norm = F.normalize(f1, p=2, dim=2, eps=1e-12)
        f2_norm = F.normalize(f2, p=2, dim=2, eps=1e-12)
        
        f1_norm = f1_norm.transpose(2, 3).unsqueeze(2) # (b, n1, 1, h*w, c)
        f2_norm = f2_norm.unsqueeze(1) # (b, 1, n2, c, h*w)

        a1 = torch.matmul(f1_norm, f2_norm) # (b, n1, n2, h*w, h*w)
        a2 = a1.transpose(3, 4) # (b, n1, n2, h*w, h*w)

        a1 = self.get_attention(a1) # (b, n1, n2, h*w)
        a2 = self.get_attention(a2) # (b, n1, n2, h*w)

        f1 = f1.unsqueeze(2) * a1.unsqueeze(3)
        f1 = f1.view(b, n1, n2, c, h, w)
        f2 = f2.unsqueeze(1) * a2.unsqueeze(3)
        f2 = f2.view(b, n1, n2, c, h, w)

        return f1.transpose(1, 2), f2.transpose(1, 2)
