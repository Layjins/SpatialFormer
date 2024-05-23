import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .activation_layer import activation_layer
from .superglue import AttentionalGNN
from .loftr import LocalFeatureTransformer

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

class CrossAttention(nn.Module):
    def __init__(self, attention_method, in_feat_c, scale_cls, norm_layer=nn.BatchNorm2d):
        super(CrossAttention, self).__init__()
        self.attention = attention_method
        self.in_feat_c = in_feat_c
        self.scale_cls = scale_cls
        self.norm_layer = norm_layer
        # cross attention module
        if self.attention == 'SuperGlue':
            # gcn attention module
            descriptor_dim = self.in_feat_c
            #GNN_layers = ['self', 'cross'] * 9
            #GNN_layers = ['self', 'cross'] * 3
            #GNN_layers = ['self', 'cross'] * 1
            GNN_layers = ['cross'] * 1
            #GNN_layers = ['cross'] * 3
            self.gnn = AttentionalGNN(descriptor_dim, GNN_layers)
        elif self.attention == 'LoFTR':
            config={
                'd_model': self.in_feat_c,
                'nhead': 8,
                #'nhead': 4,
                #'layer_names': ['self', 'cross'] * 1,
                'layer_names': ['cross'] * 1,
                'attention': 'linear'
                }  
            self.gnn = LocalFeatureTransformer(config)

    def forward(self, ftrain, ftest):
        b, num_test, num_train, c, h, w = ftrain.size()

        #### cross attention module ####
        if self.attention == 'None':
            return ftrain, ftest
        else:
            ftrain = ftrain.contiguous().view(b*num_test*num_train, c, h*w)
            ftest = ftest.contiguous().view(b*num_test*num_train, c, h*w)
            if self.attention == 'SuperGlue':
                # gnn attention
                ftrain_gnn, ftest_gnn = self.gnn(ftrain, ftest)
            elif self.attention == 'LoFTR':  
                # gnn attention
                ftrain_gnn, ftest_gnn = self.gnn(ftrain.permute(0,2,1), ftest.permute(0,2,1))
                ftrain_gnn = ftrain_gnn.permute(0,2,1)
                ftest_gnn = ftest_gnn.permute(0,2,1)

            '''
            #softmax channel-wise attention
            # ftrain
            ftrain_spatial_attention = ftrain_gnn.mean(1) # mean
            ftrain_spatial_attention = F.softmax(ftrain_spatial_attention / 0.025, dim=-1) + 1 # target attention
            ftrain_spatial_attention = ftrain_spatial_attention.unsqueeze(1)# (B, 1, N)
            ftrain_gnn = ftrain * ftrain_spatial_attention
            # ftest
            ftest_spatial_attention = ftest_gnn.mean(1) # mean
            ftest_spatial_attention = F.softmax(ftest_spatial_attention / 0.025, dim=-1) + 1 # target attention
            ftest_spatial_attention = ftest_spatial_attention.unsqueeze(1)# (B, 1, N)
            ftest_gnn = ftest * ftest_spatial_attention
            '''
            ftrain = ftrain_gnn.contiguous().view(b, num_test, num_train, c, h, w)
            ftest = ftest_gnn.contiguous().view(b, num_test, num_train, c, h, w)
        return ftrain, ftest 

class ResNetCAE(nn.Module):
    def __init__(self, args, block, layers, kernel=3, norm_layer=nn.BatchNorm2d):
        self.inplanes = 64
        self.kernel = kernel

        super(ResNetCAE, self).__init__()
        self.attention = args.attention
        self.scale_cls = args.scale_cls
        self.norm_layer = norm_layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.norm_layer(64)
        #self.relu = nn.ReLU(inplace=True)
        self.relu = activation_layer()

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1) # stride=1: out(11,11)
        self.cross_attention1 = CrossAttention(self.attention, 64, self.scale_cls, norm_layer=self.norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # stride=1: out(21,21)
        self.cross_attention2 = CrossAttention(self.attention, 128, self.scale_cls, norm_layer=self.norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.cross_attention3 = CrossAttention(self.attention, 256, self.scale_cls, norm_layer=self.norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.cross_attention4 = CrossAttention(self.attention, 512, self.scale_cls, norm_layer=self.norm_layer)

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

    def forward(self, xtrain, xtest, ytrain, ytest):
        # input shape
        # b, num_train, c_in, h_in, w_in = xtrain.size()
        # b, num_test, c_in, h_in, w_in = xtest.size()
        # output shape
        # b, n2, n1, c, h, w = ftrain.size() # n2 = num_test, n1 = ytrain.size(2)
        # b, n2, n1, c, h, w = ftest.size()

        #### stage1 ####
        # input process
        b, num_train = xtrain.size(0), xtrain.size(1)
        num_test = xtest.size(1)
        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
        x = torch.cat((xtrain, xtest), 0)

        # feature1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f = self.layer1(x)
        B, c, h, w = f.size()
        #### transfer f to ftrain and ftest ####
        ftrain = f[:b * num_train]
        ftest = f[b * num_train:]
        # feature pairs
        ftrain = ftrain.contiguous().view(b, num_train, c, -1) 
        ftest = ftest.contiguous().view(b, num_test, c, -1)
        ftrain = ftrain.unsqueeze(2).repeat(1,1,num_test,1,1).view(b, num_train, num_test, c, h, w).transpose(1, 2)
        ftest = ftest.unsqueeze(1).repeat(1,num_train,1,1,1).view(b, num_train, num_test, c, h, w).transpose(1, 2) 
        # attention1
        # b, num_test, num_train, c, h, w = ftrain.size()
        ##ftrain, ftest = self.cross_attention1(ftrain, ftest)

        #### stage2 ####
        # feature2
        b, num_test, num_train, c, h, w = ftrain.size()
        ftrain = ftrain.contiguous().view(-1, c, h, w)
        ftest = ftest.contiguous().view(-1, c, h, w)
        ftrain = self.layer2(ftrain)
        ftest = self.layer2(ftest)
        B, c, h, w = ftrain.size()
        # attention2
        ftrain = ftrain.contiguous().view(b, num_test, num_train, c, h, w)
        ftest = ftest.contiguous().view(b, num_test, num_train, c, h, w)
        ##ftrain, ftest = self.cross_attention2(ftrain, ftest)

        #### stage3 ####
        # feature3
        b, num_test, num_train, c, h, w = ftrain.size()
        ftrain = ftrain.contiguous().view(-1, c, h, w)
        ftest = ftest.contiguous().view(-1, c, h, w)
        ftrain = self.layer3(ftrain)
        ftest = self.layer3(ftest)
        B, c, h, w = ftrain.size()
        # attention3
        ftrain = ftrain.contiguous().view(b, num_test, num_train, c, h, w)
        ftest = ftest.contiguous().view(b, num_test, num_train, c, h, w)
        #ftrain, ftest = self.cross_attention3(ftrain, ftest)

        #### stage4 ####
        # feature4
        b, num_test, num_train, c, h, w = ftrain.size()
        ftrain = ftrain.contiguous().view(-1, c, h, w)
        ftest = ftest.contiguous().view(-1, c, h, w)
        ftrain = self.layer4(ftrain)
        ftest = self.layer4(ftest)
        B, c, h, w = ftrain.size()
        # attention4
        ftrain = ftrain.contiguous().view(b, num_test, num_train, c, h, w)
        ftest = ftest.contiguous().view(b, num_test, num_train, c, h, w)
        ftrain, ftest = self.cross_attention4(ftrain, ftest)

        #### stage5 ####
        # proto of ftrain and ftest
        b, num_train, K = ytrain.size()
        ytrain = ytrain.unsqueeze(2).repeat(1,1,num_test,1).view(b, num_train, num_test, K).transpose(1, 2)
        ytrain = ytrain.contiguous().view(b*num_test, num_train, -1).transpose(1, 2)
        # proto of ftrain
        ftrain = ftrain.contiguous().view(b*num_test, num_train, -1) 
        ftrain = torch.bmm(ytrain, ftrain)
        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
        ftrain = ftrain.contiguous().view(b, num_test, K, c, h, w)

        # proto of ftest
        ftest = ftest.contiguous().view(b*num_test, num_train, -1) 
        ftest = torch.bmm(ytrain, ftest)
        ftest = ftest.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftest))
        ftest = ftest.contiguous().view(b, num_test, K, c, h, w)

        return ftrain, ftest

def resnet12_cae(args, norm_layer=nn.BatchNorm2d):
    model = ResNetCAE(args, BasicBlock, [1,1,1,1], kernel=3, norm_layer=norm_layer) # resnet12
    #model = ResNetCAE(args, BasicBlock, [2,2,2,2], kernel=3, norm_layer=norm_layer) # resnet18
    return model
