import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy

from models.backbone import build_backbone, conv4, conv4_512, conv4_512_s
from models.class_center import ClassFeatConv2
from models.relation_module import distLinear, PrototypeNet, CosinNet, RelationNet, RelationNetPatch, RelationNetPatch_v1, RelationNetPatch_v2, RelationNetPatch_v3, RelationNet234, RelationNet234Patch, ClassifierNet, GlobalNet
from models.EMD import EMD_module
from .resnet12 import resnet12
from .resnet12_avg import resnet12_avg
from .resnet12_gcn import resnet12_gcn
from .resnet12_gcn_640 import resnet12_gcn_640
from .resnet12_exp import resnet12_exp_c640_k3, resnet12_exp_c640_k5, resnet12_exp_c512_k3, resnet12_exp_c768_k3, resnet12_exp_c896_k3, resnet12_exp_c960_k3, resnet12_exp_c1024_k3
from .resnet12_gcn_640_tSF import resnet12_gcn_640_tSF
from .resnet12_gcn_avg import resnet12_gcn_avg
from .resnet12_cae import resnet12_cae
from .resnet12_BDC import ResNet12_BDC
#from .backbones.rest import rest_small
#from .backbones.rest import Block as RestBlock
from .backbones.res2net_v1b import res2net50_v1b
from .backbones.wrn_mixup_model import wrn28_10
from .backbones.wrn_exp import wrn28_10_16_16_32_64, wrn28_10_32_32_64_96, wrn28_10_64_32_64_96
from .backbones.wrn_mixup_model_cam import wrn28_10_cam
from .backbones.wrn_mixup_model_gcn import wrn28_10_gcn
from .backbones.hrnet import hrnet_w18_small_v1, hrnetv2_w18
from .backbones.densenet import densenet121
from .blocks import CBAM
from .cam import CAM, ConvBlock
from .gcn import DGCN, SGCN, SCGCN, DGC_CAM, SCGCN_self, DGC_CAM2, SCGCN_Loop
from .superglue import AttentionalGNN, SelfAttentionalGNN
from .tcam import TCAM
from .cecm import CECM, CECD, CECE
from .loftr import LocalFeatureTransformer
from .nlcam import NLBlockND_CAM, NLCAM
from .position_encoding import PositionEncodingSine
from .ISDA import ISDALoss
from .vae import VAE_decoder, vae_loss_func
from .redundancy_loss import redundancy_loss_func
from .global_feat_mix_loss import GlobalFeatMixLoss
from .neck import ClassSpecific, TaskSpecific, FFN_MLP, tSF_stacker, tSF_novel, tSF_encoder, SIQ_encoder, tSF_T, tSF_T_tSF, tSF_BDC, MSF, tSF, tSF_encoder_block, tSF_tPF, tSF_plus
from .TSCA import TSCA, TSIA
from .embedding_alignment import EmbeddingAlignment


def one_hot(labels_train):
    labels_train = labels_train.cpu()
    nKnovel = 5
    labels_train_1hot_size = list(labels_train.size()) + [nKnovel,]
    labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
    labels_train_1hot = torch.zeros(labels_train_1hot_size).scatter_(len(labels_train_1hot_size) - 1, labels_train_unsqueeze, 1)
    return labels_train_1hot

class BuildCrossAttention(nn.Module):
    def __init__(self, args, attention_method, backbone_outsize, scale_cls, attention_pool=1, norm_layer=nn.BatchNorm2d):
        super(BuildCrossAttention, self).__init__()
        self.attention = attention_method
        self.CECM_mode = args.CECM_mode
        self.CECM_connect = args.CECM_connect
        self.backbone_outsize = backbone_outsize
        self.scale_cls = scale_cls
        self.attention_pool = attention_pool
        self.norm_layer = norm_layer
        # multi-class attention
        #self.conv1 = ConvBlock(1, 8, 3, s=1, p=1, norm_layer=self.norm_layer)
        #self.conv2 = nn.Conv2d(8, 1, 3, stride=1, padding=1)
        # cross attention module
        if self.attention == 'CAM':
            # cam module
            self.cam = CAM(self.backbone_outsize, norm_layer=self.norm_layer)
        elif self.attention == 'SuperGlue':
            # pos_encode (is harmful for few-shot)
            #self.pos_encoding = PositionEncodingSine(self.backbone_outsize[0])
            # gcn attention module
            descriptor_dim = self.backbone_outsize[0]
            #GNN_layers = ['self', 'cross'] * 9
            #GNN_layers = ['self', 'cross'] * 3
            #GNN_layers = ['self', 'cross'] * 1
            GNN_layers = ['cross'] * 1
            #GNN_layers = ['cross'] * 3
            self.gnn = AttentionalGNN(descriptor_dim, GNN_layers)
        elif self.attention == 'TCAM':
            # gcn attention module
            descriptor_dim = self.backbone_outsize[0]
            #GNN_layers = ['self', 'cross'] * 9
            #GNN_layers = ['self', 'cross'] * 3
            #GNN_layers = ['self', 'cross'] * 1
            GNN_layers = ['cross'] * 1
            #GNN_layers = ['cross'] * 3
            self.gnn = TCAM(descriptor_dim, GNN_layers)
        elif self.attention == 'TSIA':
            self.gnn = TSIA(self.backbone_outsize[0], num_heads=1, FFN_method='MLP', mode='TSIA')
        elif self.attention == 'TIA':
            self.gnn = TSIA(self.backbone_outsize[0], num_heads=1, FFN_method='MLP', mode='TIA')
        elif self.attention == 'CECM':
            # gcn attention module
            descriptor_dim = self.backbone_outsize[0]
            #GNN_layers = ['self', 'cross'] * 9
            #GNN_layers = ['self', 'cross'] * 3
            #GNN_layers = ['self', 'cross'] * 1
            GNN_layers = ['cross'] * 1
            #GNN_layers = ['cross'] * 3
            self.gnn = CECM(descriptor_dim, GNN_layers, mode=self.CECM_mode) # mode=['MatMul','Cosine','GCN','Transformer']
            #self.gnn = CECM(descriptor_dim, GNN_layers, mode='Transformer') # mode=['MatMul','Cosine','GCN','Transformer']
        elif self.attention == 'NLBlockND_CAM':
            # gcn attention module
            descriptor_dim = self.backbone_outsize[0]
            #GNN_layers = ['self', 'cross'] * 1
            GNN_layers = ['cross'] * 1
            self.gnn = NLBlockND_CAM(descriptor_dim, GNN_layers)
        elif self.attention == 'NLCAM':
            # gcn attention module
            descriptor_dim = self.backbone_outsize[0]
            #GNN_layers = ['self', 'cross'] * 1
            GNN_layers = ['cross'] * 1
            self.gnn = NLCAM(descriptor_dim, GNN_layers)
        elif self.attention == 'LoFTR':
            # pos_encode (is harmful for few-shot)
            #self.pos_encoding = PositionEncodingSine(self.backbone_outsize[0],temp_bug_fix=False)
            # lofter
            config={
                'd_model': self.backbone_outsize[0],
                'nhead': 8,
                #'nhead': 4,
                #'layer_names': ['self', 'cross'] * 1,
                'layer_names': ['cross'] * 1,
                'attention': 'linear'
                }  
            self.gnn = LocalFeatureTransformer(config)
        elif self.attention == 'GCN':
            # gcn attention module
            self.gcn = DGCN(self.backbone_outsize[0],
                            self.backbone_outsize[0],
                            self.backbone_outsize[1]*self.backbone_outsize[2]*2)
        elif self.attention == 'SCGCN':          
            # gcn cross-attention module
            self.gcn = SCGCN(self.backbone_outsize[0],
                            self.backbone_outsize[0],
                            self.backbone_outsize[1]*self.backbone_outsize[2]*2)
        elif self.attention == 'DGC_CAM':
            self.gcn = DGC_CAM(self.backbone_outsize[0],
                            self.backbone_outsize[1]*self.backbone_outsize[2]*2,
                            matirx_name='Mcross', scale_cls=self.scale_cls) # matrix_name={Mcorr,Mcorr_0,Mcorr_1,Mcross,Mcross_0,Mcross_1,Mself,Mself_0,Mself_1,Munit}
        elif self.attention == 'DGC_CAM2':
            self.gcn = DGC_CAM2(self.backbone_outsize[0],
                            self.backbone_outsize[1]*self.backbone_outsize[2]*2,
                            matirx_name='Mcross', scale_cls=self.scale_cls) # matrix_name={Mcorr,Mcorr_0,Mcorr_1,Mcross,Mcross_0,Mcross_1,Mself,Mself_0,Mself_1,Munit}


    def forward(self, ftrain, ftest):
        b, n1, c, h, w = ftrain.size()
        n2 = ftest.size(1)

        #### multi-class attention ####
        classes_attention = False
        if classes_attention:
            # obtain pair-feature
            ftrain_tmp = ftrain.view(b, n1, c, -1) 
            ftest_tmp = ftest.view(b, n2, c, -1)
            ftrain_tmp = ftrain_tmp.unsqueeze(2).repeat(1,1,n2,1,1).view(b, n1, n2, c, h, w).transpose(1, 2)
            ftest_tmp = ftest_tmp.unsqueeze(1).repeat(1,n1,1,1,1).view(b, n1, n2, c, h, w).transpose(1, 2)
            # similarity map
            ftrain_tmp = ftrain_tmp.mean(4)
            ftrain_tmp = ftrain_tmp.mean(4)
            ftest_norm_tmp = F.normalize(ftest_tmp, p=2, dim=3, eps=1e-12)
            ftrain_norm_tmp = F.normalize(ftrain_tmp, p=2, dim=3, eps=1e-12)
            ftrain_norm_tmp = ftrain_norm_tmp.unsqueeze(4)
            ftrain_norm_tmp = ftrain_norm_tmp.unsqueeze(5)
            similarity_map_tmp = torch.sum(ftest_norm_tmp * ftrain_norm_tmp, dim=3)
            # multi-class attention
            similarity_map_tmp = similarity_map_tmp.contiguous().view(b, n2, n1, -1)
            if not self.training:
                spatial_attention_tmp = F.softmax(similarity_map_tmp / 0.025, dim=-1) # target attention
            spatial_attention_tmp = F.softmax(F.relu(1 - similarity_map_tmp) / 0.025, dim=-1) # background attention
            #spatial_attention_tmp = F.softmax(similarity_map_tmp * 100, dim=-1)
            spatial_attention_tmp = torch.sum(spatial_attention_tmp, dim=2)
            #spatial_attention_tmp,_ = torch.max(spatial_attention_tmp, dim=2)
            spatial_attention_tmp = spatial_attention_tmp.contiguous().view(b, n2, h, w)
            spatial_attention_tmp = spatial_attention_tmp.unsqueeze(2)
            #spatial_attention_tmp = spatial_attention_tmp.contiguous().view(b * n2, -1, h, w)
            #spatial_attention_tmp = F.relu(self.conv1(spatial_attention_tmp))
            #spatial_attention_tmp = self.conv2(spatial_attention_tmp)
            #spatial_attention_tmp = spatial_attention_tmp.contiguous().view(b, n2, -1, h, w)
            ftest = ftest * (spatial_attention_tmp + 1) #* (spatial_attention_tmp + 1)

        #### cross attention module ####
        if self.attention == 'None':
            #b, n1, c, h, w = ftrain.size()
            #n2 = ftest.size(1)
            ftrain = ftrain.view(b, n1, c, -1) 
            ftest = ftest.view(b, n2, c, -1)
            ftrain = ftrain.unsqueeze(2).repeat(1,1,n2,1,1).view(b, n1, n2, c, h, w).transpose(1, 2)
            ftest = ftest.unsqueeze(1).repeat(1,n1,1,1,1).view(b, n1, n2, c, h, w).transpose(1, 2)
            #print(ftrain.shape) # torch.Size([4, 75, 5, 512, 6, 6])
            #print(ftest.shape) # torch.Size([4, 75, 5, 512, 6, 6])
        elif self.attention == 'CAM':
            # CAM module
            ftrain, ftest = self.cam(ftrain, ftest)
            #print(ftrain.shape) # torch.Size([4, 75, 5, 512, 6, 6])
            #print(ftest.shape) # torch.Size([4, 75, 5, 512, 6, 6])   
        else:
            # concat pair features
            #b, n1, c, h, w = ftrain.size()
            #n2 = ftest.size(1)
            ftrain = ftrain.view(b, n1, c, -1) 
            ftest = ftest.view(b, n2, c, -1)
            ftrain = ftrain.unsqueeze(2).repeat(1,1,n2,1,1).view(b, n1, n2, c, h, w).transpose(1, 2)
            ftest = ftest.unsqueeze(1).repeat(1,n1,1,1,1).view(b, n1, n2, c, h, w).transpose(1, 2)
            #print(ftrain.shape) # torch.Size([4, 75, 5, 512, 6, 6])
            #print(ftest.shape) # torch.Size([4, 75, 5, 512, 6, 6])
            ftrain = ftrain.contiguous().view(b, n2, n1, c, h*w)
            ftest = ftest.contiguous().view(b, n2, n1, c, h*w)
            f_cat = torch.cat((ftrain, ftest), 4).contiguous().view(b*n2*n1, c, h*w*2)

            if self.attention == 'DGC_CAM':
                # GCN cross-attention module
                f_cat = self.gcn(f_cat)
                # split features
                f_cat = f_cat.contiguous().view(b, n2, n1, c, h*w*2)
                ftrain = f_cat[:,:,:,:,:h*w]
                ftest = f_cat[:,:,:,:,h*w:]
                ftrain = ftrain.contiguous().view(b, n2, n1, c, h, w)
                ftest = ftest.contiguous().view(b, n2, n1, c, h, w)
            elif self.attention == 'NLCAM':
                ftrain = ftrain.contiguous().view(b*n2*n1, c, h, w)
                ftest = ftest.contiguous().view(b*n2*n1, c, h, w)
                # gnn attention
                ftrain, ftest = self.gnn(ftrain, ftest)

                ftrain = ftrain.contiguous().view(b, n2, n1, c, h, w)
                ftest = ftest.contiguous().view(b, n2, n1, c, h, w)
            elif self.attention == 'TCAM':
                ftrain = ftrain.contiguous().view(b*n2*n1, c, h*w)
                ftest = ftest.contiguous().view(b*n2*n1, c, h*w)
                # gnn attention
                ftrain, ftest = self.gnn(ftrain, ftest)

                ftrain = ftrain.contiguous().view(b, n2, n1, c, h, w)
                ftest = ftest.contiguous().view(b, n2, n1, c, h, w)   
            elif (self.attention == 'TSIA') or (self.attention == 'TIA'):
                ftrain = ftrain.contiguous().view(b*n2*n1, c, h, w)
                ftest = ftest.contiguous().view(b*n2*n1, c, h, w)
                # gnn attention
                ftrain, ftest = self.gnn(ftrain, ftest)

                ftrain = ftrain.contiguous().view(b, n2, n1, c, h, w)
                ftest = ftest.contiguous().view(b, n2, n1, c, h, w)   
            elif self.attention == 'CECM':
                if self.CECM_connect == 'one2many':
                    ftrain = ftrain.contiguous().view(b*n2, n1, c*h*w).transpose(1, 2)
                    ftrain = ftrain.contiguous().view(b*n2, c, h*w*n1)
                    ftest = ftest.transpose(1, 2)
                    ftest = ftest[:,:1,:,:,:]
                    ftest = ftest.contiguous().view(b*n2, c, h*w)
                    # gnn attention
                    ftrain, ftest, relation_map_ftrain, relation_map_ftest = self.gnn(ftrain, ftest)

                    ftrain = ftrain.contiguous().view(b*n2, c*h*w, n1).transpose(1, 2)
                    ftrain = ftrain.contiguous().view(b, n2, n1, c, h, w)
                    ftest = ftest.contiguous().view(b, n2, c, h*w)
                    ftest = ftest.unsqueeze(1).repeat(1,n1,1,1,1).view(b, n1, n2, c, h, w).transpose(1, 2)
                    relation_map_ftrain = relation_map_ftrain.contiguous().view(b*n2, h*w, n1).transpose(1, 2)
                    relation_map_ftrain = relation_map_ftrain.contiguous().view(b, n2, n1, h, w)
                    relation_map_ftest = relation_map_ftest.contiguous().view(b, n2, h, w)
                    relation_map_ftest = relation_map_ftest.unsqueeze(2).repeat(1,1,n1,1,1)
                else: # one2one
                    ftrain = ftrain.contiguous().view(b*n2*n1, c, h*w)
                    ftest = ftest.contiguous().view(b*n2*n1, c, h*w)
                    # gnn attention
                    ftrain, ftest, relation_map_ftrain, relation_map_ftest = self.gnn(ftrain, ftest)

                    ftrain = ftrain.contiguous().view(b, n2, n1, c, h, w)
                    ftest = ftest.contiguous().view(b, n2, n1, c, h, w)   
                    relation_map_ftrain = relation_map_ftrain.contiguous().view(b, n2, n1, h, w)
                    relation_map_ftest = relation_map_ftest.contiguous().view(b, n2, n1, h, w) 
            else:
                # pos_encode (is harmful for few-shot)
                # ftrain = ftrain.contiguous().view(b*n2*n1, c, h, w)
                # ftest = ftest.contiguous().view(b*n2*n1, c, h, w)
                # ftrain = self.pos_encoding(ftrain)
                # ftest = self.pos_encoding(ftest)
                ftrain = ftrain.contiguous().view(b*n2*n1, c, h*w)
                ftest = ftest.contiguous().view(b*n2*n1, c, h*w)
                # attention
                if self.attention == 'SuperGlue':
                    # gnn attention
                    ftrain_gnn, ftest_gnn = self.gnn(ftrain, ftest)
                elif self.attention == 'NLBlockND_CAM':
                    ftrain = ftrain.contiguous().view(b*n2*n1, c, h, w)
                    ftest = ftest.contiguous().view(b*n2*n1, c, h, w)
                    # gnn attention
                    ftrain_gnn, ftest_gnn = self.gnn(ftrain, ftest)

                    ftrain_gnn = ftrain_gnn.contiguous().view(b*n2*n1, c, h*w)
                    ftest_gnn = ftest_gnn.contiguous().view(b*n2*n1, c, h*w)
                    ftrain = ftrain.contiguous().view(b*n2*n1, c, h*w)
                    ftest = ftest.contiguous().view(b*n2*n1, c, h*w)
                elif self.attention == 'LoFTR':
                    # gnn attention
                    ftrain_gnn, ftest_gnn = self.gnn(ftrain.permute(0,2,1), ftest.permute(0,2,1))
                    ftrain_gnn = ftrain_gnn.permute(0,2,1)
                    ftest_gnn = ftest_gnn.permute(0,2,1)
                #'''
                #softmax spatial-wise attention
                # ftrain
                ftrain_spatial_attention = ftrain_gnn.mean(1) # mean
                ftrain_spatial_attention = F.softmax(ftrain_spatial_attention / 0.025, dim=-1) + 1
                ftrain_spatial_attention = ftrain_spatial_attention.unsqueeze(1)# (B, 1, N)
                ftrain_gnn = ftrain * ftrain_spatial_attention
                # ftest
                ftest_spatial_attention = ftest_gnn.mean(1) # mean
                ftest_spatial_attention = F.softmax(ftest_spatial_attention / 0.025, dim=-1) + 1
                ftest_spatial_attention = ftest_spatial_attention.unsqueeze(1)# (B, 1, N)
                ftest_gnn = ftest * ftest_spatial_attention
                #'''
                ftrain = ftrain_gnn.contiguous().view(b, n2, n1, c, h, w)
                ftest = ftest_gnn.contiguous().view(b, n2, n1, c, h, w)    

        # ouput cross attention results
        attention_results={
            'ftrain': None,
            'ftest': None,
            'relation_map_ftrain': None,
            'relation_map_ftest': None
        }
        # attention pool
        if self.attention_pool > 1:
            b, n2, n1, c, h, w = ftrain.size()
            ftrain = ftrain.contiguous().view(b*n2*n1, c, h, w)
            ftest = ftest.contiguous().view(b*n2*n1, c, h, w)
            ##ftrain = F.max_pool2d(ftrain,kernel_size=self.attention_pool)
            ##ftest = F.max_pool2d(ftest,kernel_size=self.attention_pool)
            ftrain = F.avg_pool2d(ftrain,kernel_size=self.attention_pool)
            ftest = F.avg_pool2d(ftest,kernel_size=self.attention_pool)
            b_n2_n1, c, h, w = ftrain.size()
            ftrain = ftrain.contiguous().view(b, n2, n1, c, h, w)
            ftest = ftest.contiguous().view(b, n2, n1, c, h, w)
        
        attention_results['ftrain']=ftrain
        attention_results['ftest']=ftest
        if self.attention == 'CECM':
            attention_results['relation_map_ftrain']=relation_map_ftrain
            attention_results['relation_map_ftest']=relation_map_ftest
        return attention_results

class SimilarityResFunc(nn.Module):
    def __init__(self, args, method, attention_method, backbone_outsize, scale_cls, norm_layer=nn.BatchNorm2d, embed_classifier_train=False, embed_classifier_test=False, cluster_embed_by_support=False, embed_classifier_weight=0.05):
        super(SimilarityResFunc, self).__init__()  
        self.method = method
        self.backbone = args.backbone
        self.attention_method = attention_method
        self.backbone_outsize = backbone_outsize
        self.scale_cls = scale_cls
        self.norm_layer = norm_layer
        self.num_classes = args.num_classes
        self.nExemplars = args.nExemplars
        self.nKnovel = args.nKnovel
        self.auxiliary_attention = args.auxiliary_attention
        self.emd_metric = args.emd_metric
        self.task_specific_scaling = args.task_specific_scaling
        self.adaptive_metrics = args.adaptive_metrics
        self.distance_metric = args.distance_metric
        self.CECD_mode = args.CECD_mode
        self.embed_classifier_train = embed_classifier_train
        self.embed_classifier_test = embed_classifier_test
        self.cluster_embed_by_support = cluster_embed_by_support
        self.embed_classifier_weight = embed_classifier_weight
        self.neck = args.neck
        self.tSF_plus_mode = args.tSF_plus_mode
        self.num_queries = args.num_queries

        if self.distance_metric == 'CECD':
            descriptor_dim = self.backbone_outsize[0]
            GNN_layers = ['cross'] * 1
            self.distance_func = CECD(descriptor_dim, GNN_layers, mode=self.CECD_mode, scale_cls=self.scale_cls) # mode=['MatMul','Cosine','GCN','Transformer']    
        # attention for global and rotation classification
        if self.auxiliary_attention:
            self.attention_gcn = SCGCN_self(self.backbone_outsize[0],
                self.backbone_outsize[1]*self.backbone_outsize[2])
        # global loss
        self.clasifier = nn.Conv2d(self.backbone_outsize[0], self.num_classes, kernel_size=1)
        # rotation loss
        self.rotation_loss = args.rotation_loss
        if self.rotation_loss:
            self.rotate_classifier = nn.Conv2d(self.backbone_outsize[0], 4, kernel_size=1)
        # GlobalFeatMixLoss
        self.global_feat_mix_loss = args.global_feat_mix_loss
        if self.global_feat_mix_loss:
            self.GlobalFeatMixLoss = GlobalFeatMixLoss(self.backbone_outsize, self.num_classes, mix_alpha=2.0, using_focal_loss=args.using_focal_loss)
        # isda loss
        self.isda_loss = args.isda_loss
        if self.isda_loss:
            self.isda_criterion = ISDALoss(self.backbone_outsize[0], self.num_classes)
            self.isda_avgpool = nn.AvgPool2d(self.backbone_outsize[1], stride=1)
            self.isda_classifier = nn.Linear(self.backbone_outsize[0], self.num_classes)
        # redundancy loss
        self.redundancy_loss = args.redundancy_loss

        if self.method in ['RelationNet', 'RelationNetPlus', 'IMLN']:
            # metric weight
            self.relation_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
            self.proto_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
            self.cosin_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
            # self.relation_weight = 1 + torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
            # self.proto_weight = 1 + torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
            # self.cosin_weight = 1 + torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
            if self.adaptive_metrics:
                self.adaptive_relation_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
                self.adaptive_proto_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
                self.adaptive_cosin_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=False)
            # relation module
            metric_scale=False # for ProtoNetE and ProtoNetC
            if self.method == 'RelationNet':
                #self.relation_module = RelationNet(self.backbone_outsize, self.backbone_outsize[0], args.relation_dim, metric_scale=False)
                self.relation_module = RelationNetPatch(self.backbone_outsize, self.backbone_outsize[0], args.relation_dim, metric_scale=False)
            else:
                #self.relation_module = RelationNet234(self.backbone_outsize, self.backbone_outsize[0], args.relation_dim, metric_scale=False)
                #self.relation_module = RelationNet234Patch(self.backbone_outsize, self.backbone_outsize[0], args.relation_dim, metric_scale=False)
                self.relation_module = RelationNetPatch(self.backbone_outsize, self.backbone_outsize[0], args.relation_dim, metric_scale=False)
                #self.relation_module = RelationNetPatch_v1(self.backbone_outsize, self.backbone_outsize[0], args.relation_dim, metric_scale=False)
                #self.relation_module = RelationNetPatch_v2(self.backbone_outsize, self.backbone_outsize[0], args.relation_dim, metric_scale=False)
                #self.relation_module = RelationNetPatch_v3(self.backbone_outsize, self.backbone_outsize[0], args.relation_dim, metric_scale=False)
            self.prototype_net = PrototypeNet(metric_scale)
            self.cosin_net = CosinNet(metric_scale)
            if self.emd_metric:
                self.emd_net = EMD_module()
            self.scale_proto = self.scale_cls
            #self.scale_proto = 5

    def test(self, ftrain, ftest, embed_classifier=False, cluster_embed_by_support=False, query_embed=None):
        if embed_classifier:
            b, n2, n1, c, h, w = ftrain.size()
            ftrain_embed = ftrain.contiguous().view(b * n2 * n1, c, h, w)
            ftest_embed = ftest.contiguous().view(b * n2 * n1, c, h, w)
            # classifier
            if query_embed == None:
                train_score = self.clasifier(ftrain_embed) # (b * n2 * n1, base_class_num, h, w)
                test_score = self.clasifier(ftest_embed) # (b * n2 * n1, base_class_num, h, w)
            else:
                ftrain_embed = ftrain.contiguous().view(b * n2 * n1, c, h * w)
                ftest_embed = ftest.contiguous().view(b * n2 * n1, c, h * w)
                # query_embed = (num_queries, c)
                query_embed_weight = query_embed.weight
                query_embed_weight = query_embed_weight.transpose(0,1).unsqueeze(0) # (1, c, num_queries)
                train_score = torch.matmul(ftrain_embed.transpose(1,2), query_embed_weight) # (b * n2 * n1, h * w, num_queries)
                test_score = torch.matmul(ftest_embed.transpose(1,2), query_embed_weight) # (b * n2 * n1, h * w, num_queries)
                train_score = train_score.transpose(1,2).contiguous().view(b * n2 * n1, -1, h, w) # (b * n2 * n1, num_queries, h, w)
                test_score = test_score.transpose(1,2).contiguous().view(b * n2 * n1, -1, h, w) # (b * n2 * n1, num_queries, h, w)
            # hw mean
            train_score = train_score.view(*train_score.size()[:2], -1).mean(2) # (b * n2 * n1, base_class_num)
            test_score = test_score.view(*test_score.size()[:2], -1).mean(2)
            # cluster_embed_by_support: reduce (b * n2 * n1, base_class_num) to  (b * n2 * n1, n1)
            if cluster_embed_by_support:
                train_score = train_score.contiguous().view(b * n2, n1, -1)
                test_score = test_score.contiguous().view(b * n2, n1, -1)
                train_score_trans = train_score.transpose(1,2)  # (b * n2, base_class_num, n1)
                support_cluster = (train_score_trans == train_score_trans.max(dim=-1, keepdim=True)[0]).to(dtype=torch.float32)  # (b * n2, base_class_num, n1)
                train_score = torch.matmul(train_score, support_cluster) # (b * n2, n1, n1)
                test_score = torch.matmul(test_score, support_cluster) # (b * n2, n1, n1)
                train_score = train_score.contiguous().view(b * n2 * n1, n1)
                test_score = test_score.contiguous().view(b * n2 * n1, n1)

            # cosine
            train_score = F.normalize(train_score, p=2, dim=train_score.dim()-1, eps=1e-12)
            test_score = F.normalize(test_score, p=2, dim=test_score.dim()-1, eps=1e-12)
            embed_classifier_scores = self.scale_cls * torch.sum(test_score * train_score, dim=-1) # (b * n2 * n1)
            embed_classifier_scores = embed_classifier_scores.contiguous().view(b, n2, n1)

        # cosine classifier
        ftrain = ftrain.view(*ftrain.size()[:4], -1).mean(4)
        ftest = ftest.view(*ftest.size()[:4], -1).mean(4)
        #ftrain = ftrain.mean(4)
        #ftrain = ftrain.mean(4)
        #ftest = ftest.mean(4)
        #ftest = ftest.mean(4)

        # task_specific_scaling
        if self.task_specific_scaling:
            ftrain_center = ftrain.mean(2) # (b,n2,c)
            ftest_center = ftest.mean(2) # (b,n2,c)
            task_center = ((self.nKnovel * self.nExemplars) * ftrain_center + ftest_center) / (self.nKnovel * self.nExemplars + 1)
            task_specific_scale = task_center.mean(2) # (b,n2)
            task_specific_scale = task_specific_scale.unsqueeze(2).unsqueeze(3) # (b,n2,1,1)
            # task_specific_scaling
            ftrain = ftrain - task_specific_scale
            ftest = ftest - task_specific_scale
        '''
        # centering
        ftrain_center = ftrain.mean(2) # (b,n2,c)
        ftrain_center = ftrain_center.unsqueeze(2) # (b,n2,1,c)
        ftest  = 2*ftest - ftrain_center # centering3
        ftrain  = 2*ftrain - ftrain_center # centering3
        '''
        '''
        # NCC
        ftest_mean = ftest.mean(3).unsqueeze(3) # (b,n2,n1,1)
        ftest = ftest - ftest_mean
        ftrain_mean = ftrain.mean(3).unsqueeze(3) # (b,n2,n1,1)
        ftrain = ftrain - ftrain_mean
        '''
        ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
        scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1)
        #print(scores.shape) # torch.Size([4, 75, 5])
        if embed_classifier:
            scores = scores + self.embed_classifier_weight * embed_classifier_scores
        return scores

    def forward(self, ytest, attention_results, backbone_name=None,pids=None, class_center='MeanWise', isda_ratio=0, adapt_metrics=False, finetune_novel_query=False, query_embed=None):
        ##### output results init####
        output_results={
            'ytest': None,
            'cls_scores': None,
            'embed_classifier_scores': None,
            'rotate_scores': None,
            'global_feat_mix_loss_res': None,
            'redundancy_loss_res': None,
            'vae_loss_res': None,
            'isda_loss_res': None,
            'cos_scores_patch': None,
            'prototype_patch': None,
            'relation_scores_patch': None,
            'metric_scores_patch': None,
            'emd_scores': None,
            'metric_scores': None
        }  

        ftrain = attention_results['ftrain']
        ftest = attention_results['ftest']
        b, n2, n1, c, h, w = ftrain.size()
        # pair-shift
        # pair_shift = ((ftrain + ftest)/2).mean(3).unsqueeze(3)
        # ftrain = F.relu(ftrain - pair_shift)
        # ftest = F.relu(ftest - pair_shift)
        # dynamic center
        #ftrain = (ftrain*self.nExemplars + ftest)/(self.nExemplars + 1) # mean
        #ftrain = (ftrain + ftest)/2 # avg
        # unit_spatial
        #ftrain = F.softmax(ftrain, dim=3)
        #ftest = F.softmax(ftest, dim=3)
        if self.method == 'CAN':
            # testing
            if self.distance_metric == 'CECD':
                ftrain_CECD = ftrain.contiguous().view(b*n2*n1, c, h*w)
                ftest_CECD = ftest.contiguous().view(b*n2*n1, c, h*w)
                cls_scores = self.distance_func(ftrain_CECD, ftest_CECD) # (b*n2*n1, h*w)
                cls_scores = cls_scores.contiguous().view(b*n2, n1, h, w)
                if not self.training and (backbone_name == None) and (finetune_novel_query == False):
                    CECD_scores = cls_scores.contiguous().view(b, n2, n1, h*w) # torch.Size([4, 75, 5])
                    CECD_scores = CECD_scores.mean(3)
                    ''' global scores
                    global_scores = self.test(ftrain, ftest) # torch.Size([4, 75, 5])
                    CECD_scores = CECD_scores + global_scores
                    '''
                    output_results['metric_scores'] = CECD_scores
                    return output_results
                # CAN similarity losses
                output_results['cls_scores'] = cls_scores
            else: # Cosine
                if not self.training and (backbone_name == None) and (finetune_novel_query == False):
                    #return self.test(ftrain, ftest) # torch.Size([4, 75, 5])
                    global_scores = self.test(ftrain, ftest, embed_classifier=self.embed_classifier_test, cluster_embed_by_support=self.cluster_embed_by_support, query_embed=query_embed) # torch.Size([4, 75, 5])
                    output_results['metric_scores'] = global_scores
                    return output_results
                # training
                ##### CAN loss ####
                # embed_classifier loss for cosine similarity
                if self.embed_classifier_train:
                    # b, n2, n1, c, h, w = ftrain.size()
                    ftrain_embed = ftrain.contiguous().view(b * n2 * n1, c, h, w)
                    ftest_embed = ftest.contiguous().view(b * n2 * n1, c, h, w)
                    # classifier
                    if query_embed == None:
                        train_score = self.clasifier(ftrain_embed) # (b * n2 * n1, base_class_num, h, w)
                        test_score = self.clasifier(ftest_embed) # (b * n2 * n1, base_class_num, h, w)
                    else:
                        ftrain_embed = ftrain.contiguous().view(b * n2 * n1, c, h * w)
                        ftest_embed = ftest.contiguous().view(b * n2 * n1, c, h * w)
                        # query_embed = (num_queries, c)
                        query_embed_weight = query_embed.weight
                        query_embed_weight = query_embed_weight.transpose(0,1).unsqueeze(0) # (1, c, num_queries)
                        train_score = torch.matmul(ftrain_embed.transpose(1,2), query_embed_weight) # (b * n2 * n1, h * w, num_queries)
                        test_score = torch.matmul(ftest_embed.transpose(1,2), query_embed_weight) # (b * n2 * n1, h * w, num_queries)
                        train_score = train_score.transpose(1,2).contiguous().view(b * n2 * n1, -1, h, w) # (b * n2 * n1, num_queries, h, w)
                        test_score = test_score.transpose(1,2).contiguous().view(b * n2 * n1, -1, h, w) # (b * n2 * n1, num_queries, h, w)
                    # cosine
                    train_score = train_score.view(*train_score.size()[:2], -1).mean(2) # (b * n2 * n1, base_class_num)
                    train_score = F.normalize(train_score, p=2, dim=train_score.dim()-1, eps=1e-12)
                    test_score = F.normalize(test_score, p=2, dim=test_score.dim()-1, eps=1e-12) # (b * n2 * n1, base_class_num, h, w)
                    train_score = train_score.unsqueeze(2)
                    train_score = train_score.unsqueeze(3) # (b * n2 * n1, base_class_num, 1, 1)
                    embed_classifier_scores = self.scale_cls * torch.sum(test_score * train_score, dim=1) # (b * n2 * n1, h, w)
                    embed_classifier_scores = embed_classifier_scores.contiguous().view(b * n2, n1, h, w)
                    output_results['embed_classifier_scores'] = embed_classifier_scores
                # cos similarity (patch loss)
                ftrain = ftrain.view(*ftrain.size()[:4], -1).mean(4)
                #ftrain = ftrain.mean(4)
                #ftrain = ftrain.mean(4)
                '''
                # centering
                ftrain_center = ftrain.mean(2) # (b,n2,c)
                ftrain_center = ftrain_center.unsqueeze(2).repeat(1,1,n1,1) # (b,n2,1,c)
                ftrain  = 2*ftrain - ftrain_center # centering3
                ftrain_center = ftrain_center.unsqueeze(4)
                ftrain_center = ftrain_center.unsqueeze(5)
                ftest_centering  = 2*ftest - ftrain_center # centering3
                ''' 
                '''
                # NCC
                ftest_mean = ftest.mean(3).unsqueeze(3) # (b,n2,n1,1,h,w)
                ftest = ftest - ftest_mean
                ftrain_mean = ftrain.mean(3).unsqueeze(3) # (b,n2,n1,1)
                ftrain = ftrain - ftrain_mean
                '''
                ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)
                ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
                ftrain_norm = ftrain_norm.unsqueeze(4)
                ftrain_norm = ftrain_norm.unsqueeze(5)
                #print(ftest_norm.shape) # torch.Size([4, 75, 5, 512, 6, 6])
                #print(ftrain_norm.shape) # torch.Size([4, 75, 5, 512, 1, 1])
                cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm, dim=3)
                #print(cls_scores.shape) # torch.Size([4, 75, 5, 6, 6])
                cls_scores = cls_scores.view(b * n2, *cls_scores.size()[2:])
                #print(cls_scores.shape) # torch.Size([300, 5, 6, 6])
                #_, cls_scores_map = torch.max(cls_scores, 1) # (b*n2,h,w)
                #print("cls_scores_map")
                #print(cls_scores_map[0])
                # CAN similarity losses
                output_results['cls_scores'] = cls_scores
                '''
                # testing : global+local
                if not self.training and (backbone_name == None) and (finetune_novel_query == False):
                    cls_scores_patch = cls_scores
                    cls_scores = cls_scores.view(b*n2, n1, h*w)
                    cls_scores = F.softmax(cls_scores, dim=1) # (b*n2,n1,h*w)
                    cls_scores = torch.sum(cls_scores, dim=2)
                    cls_scores = cls_scores.view(b, n2, n1)
                    #print(cls_scores)
                    #print(global_scores)
                    cls_scores = cls_scores + global_scores # global+local
                    #cls_scores = global_scores # global
                    output_results['metric_scores'] = cls_scores
                    #return cls_scores
                    return output_results
                '''
            ##### output results loss####   
            # other losses
            if class_center == 'MeanWise':
                if self.isda_loss:
                    _, isda_y = torch.max(ytest.view(-1,K), 1)
                # global loss
                ftest = ftest.contiguous().view(b, n2, n1, -1)
                ftest = ftest.transpose(2, 3) 
                ytest = ytest.unsqueeze(3)
                ftest = torch.matmul(ftest, ytest)
                # attention for ftest
                if self.auxiliary_attention:
                    ftest = ftest.view(b * n2, -1, h*w)
                    ftest = self.attention_gcn(ftest)
                ftest = ftest.view(b * n2, -1, h, w) 
                ytest = self.clasifier(ftest)
                #print(ytest.shape) # torch.Size([300, 64, 6, 6]) # patch loss
                #print(cls_scores.shape) # torch.Size([300, 5, 6, 6]) # patch loss
                #print(cls_scores[0])
                if self.rotation_loss:
                    # rotation loss
                    rotate_scores = self.rotate_classifier(ftest)
                # GlobalFeatMixLoss
                if self.global_feat_mix_loss:
                    global_feat_mix_loss_res = self.GlobalFeatMixLoss(ftest, pids)
                # isda loss
                if self.isda_loss:
                    isda_feature = self.isda_avgpool(ftest)
                    isda_feature = isda_feature.view(isda_feature.size(0), -1)
                    isda_loss_res, isda_y_res = self.isda_criterion(isda_feature, self.isda_classifier, isda_y, isda_ratio)
                if self.redundancy_loss:
                    redundancy_loss_res = redundancy_loss_func(ftest)

                ##### output results loss####
                # CAN global losses
                output_results['ytest'] = ytest
                # output_results['metric_scores'] = ytest.mean(-1).mean(-1)
                # other losses
                if self.rotation_loss:
                    output_results['rotate_scores'] = rotate_scores
                if self.global_feat_mix_loss:
                    output_results['global_feat_mix_loss_res'] = global_feat_mix_loss_res
                if self.isda_loss:
                    output_results['isda_loss_res'] = isda_loss_res
                if self.redundancy_loss:
                    output_results['redundancy_loss_res'] = redundancy_loss_res
            return output_results
        elif self.method in ['RelationNet', 'RelationNetPlus', 'IMLN']:
            #### RelationNet, RelationNetPlus, IMLN ####
            #print(self.cosin_weight)
            #print(self.proto_weight)
            #print(self.relation_weight)
            if self.emd_metric:
                # EMD scores
                emd_scores = self.scale_cls * self.emd_net(ftrain, ftest).view(-1,n1)
            # testing
            if (not self.training and (backbone_name == None) and (finetune_novel_query == False)) or self.adaptive_metrics:
                # cos scores
                cos_scores = self.test(ftrain, ftest).view(-1,n1)
                # PrototypeNet scores
                ftrain_mean = ftrain.mean(4)
                ftrain_mean = ftrain_mean.mean(4)
                ftest_mean = ftest.mean(4)
                ftest_mean = ftest_mean.mean(4)
                ftest_mean = F.normalize(ftest_mean, p=2, dim=ftest_mean.dim()-1, eps=1e-12) # norm
                ftrain_mean = F.normalize(ftrain_mean, p=2, dim=ftrain_mean.dim()-1, eps=1e-12) # norm
                prototype = self.scale_proto * self.prototype_net(ftrain_mean.contiguous().view(b*n2, n1, -1), ftest_mean.contiguous().view(b*n2, n1, -1))
                # RelationNet scores
                relation_pairs = torch.cat((ftrain_mean.unsqueeze(4).unsqueeze(5),ftest_mean.unsqueeze(4).unsqueeze(5)),3).contiguous().view(-1,c*2,1,1)
                relation_scores = self.relation_module(relation_pairs.contiguous()).contiguous().view(-1,n1) #torch.Size([300, 5])

                logsoftmax = nn.LogSoftmax(dim=1)
                if self.adaptive_metrics:
                    #global_scores = self.adaptive_relation_weight*(1+self.cosin_weight)*logsoftmax(cos_scores) + self.adaptive_proto_weight*(1+self.proto_weight)*logsoftmax(torch.sigmoid(prototype)) + self.adaptive_cosin_weight*(1+self.relation_weight)*logsoftmax(torch.sigmoid(relation_scores))
                    global_scores = (1+self.adaptive_relation_weight)*logsoftmax(cos_scores) + (1+self.adaptive_proto_weight)*logsoftmax(torch.sigmoid(prototype)) + (1+self.adaptive_cosin_weight)*logsoftmax(torch.sigmoid(relation_scores))
                    if self.training and adapt_metrics:
                        output_results['metric_scores'] = global_scores
                        return output_results
                else:
                    ##global_scores = self.cosin_weight*logsoftmax(cos_scores) + self.proto_weight*logsoftmax(torch.sigmoid(prototype)) + self.relation_weight*logsoftmax(torch.sigmoid(relation_scores))
                    #print("test: cosin_weight={},proto_weight={},relation_weight={}".format(self.cosin_weight.data, self.proto_weight.data, self.relation_weight.data))
                    global_scores = (1+self.cosin_weight)*logsoftmax(cos_scores) + (1+self.proto_weight)*logsoftmax(torch.sigmoid(prototype)) + (1+self.relation_weight)*logsoftmax(torch.sigmoid(relation_scores))
                
                output_results['metric_scores'] = global_scores  
                #output_results['metric_scores'] = F.softmax(relation_scores,dim=1)
                #output_results['metric_scores'] = F.softmax(prototype,dim=1)
                #output_results['metric_scores'] = F.softmax(cos_scores,dim=1)
                #output_results['metric_scores'] = F.softmax(emd_scores,dim=1)
                #return output_results
            # training
            # cos similarity (patch loss)
            ftrain = ftrain.mean(4)
            ftrain = ftrain.mean(4)
            ftest_norm = F.normalize(ftest, p=2, dim=3, eps=1e-12)
            ftrain_norm = F.normalize(ftrain, p=2, dim=3, eps=1e-12)
            ftrain_norm = ftrain_norm.unsqueeze(4)
            ftrain_norm = ftrain_norm.unsqueeze(5)
            cos_scores_patch = self.scale_cls * torch.sum(ftest_norm * ftrain_norm, dim=3)
            cos_scores_patch = cos_scores_patch.contiguous().view(b * n2, *cos_scores_patch.size()[2:])
            #cos_scores_patch = torch.rand(b*n2,n1,h,w).cuda()
            # testing
            if not self.training and (backbone_name == None) and (finetune_novel_query == False):
                if 'conv4' in self.backbone:
                    cls_scores = global_scores
                else:
                    # cos scores
                    cos_scores_patch = cos_scores_patch.contiguous().view(b*n2, n1, h*w)
                    cos_scores_patch = F.softmax(cos_scores_patch, dim=1) # (b*n2,n1,h*w)
                    cos_scores_patch = torch.sum(cos_scores_patch, dim=2)
                    logsoftmax = nn.LogSoftmax(dim=1)
                    cls_scores = logsoftmax(cos_scores_patch)
                    cls_scores = cls_scores + global_scores
                if self.emd_metric:
                    cls_scores = cls_scores + logsoftmax(emd_scores)
                output_results['metric_scores'] = cls_scores
                return output_results
            # euclidean similarity (patch loss)
            ftrain_e = ftrain_norm.squeeze(5) # norm
            ##ftrain_e = ftrain.unsqueeze(4)
            ftrain_e = ftrain_e.repeat(1,1,1,1,h*w).transpose(3, 4).contiguous().view(b*n2*n1, h*w, c)
            ftest_e = ftest_norm # norm
            ##ftest_e = ftest
            ftest_e = ftest_e.contiguous().view(b,n2,n1,c,h*w).transpose(3, 4).contiguous().view(b*n2*n1, h*w, c)
            prototype_patch = self.scale_proto * self.prototype_net(ftrain_e,ftest_e).contiguous().view(b*n2,n1,h,w)
            #prototype_patch = torch.rand(b*n2,n1,h,w).cuda()
            # relation similarity (patch loss)
            relation_pairs_patch = torch.cat((ftrain_e,ftest_e),2).contiguous().view(-1,c*2).unsqueeze(2).unsqueeze(3)
            relation_pairs_patch = relation_pairs_patch.contiguous()
            relation_scores_patch = self.relation_module(relation_pairs_patch).contiguous().view(b*n2,n1,h,w)
            #relation_scores_patch = torch.rand(b*n2,n1,h,w).cuda()

            ##### output results loss####
            # IMLN losses
            output_results['cos_scores_patch'] = cos_scores_patch
            output_results['prototype_patch'] = prototype_patch
            output_results['relation_scores_patch'] = relation_scores_patch
            #if backbone_name != None:
            logsoftmax = nn.LogSoftmax(dim=1)
            if self.adaptive_metrics:
                global_scores_patch = (1+self.adaptive_relation_weight)*logsoftmax(cos_scores_patch) + (1+self.adaptive_proto_weight)*logsoftmax(torch.sigmoid(prototype_patch)) + (1+self.adaptive_cosin_weight)*logsoftmax(torch.sigmoid(relation_scores_patch))
            else:
                global_scores_patch = (1+self.cosin_weight)*logsoftmax(cos_scores_patch) + (1+self.proto_weight)*logsoftmax(torch.sigmoid(prototype_patch)) + (1+self.relation_weight)*logsoftmax(torch.sigmoid(relation_scores_patch))
            output_results['metric_scores_patch'] = global_scores_patch  
            if self.emd_metric:
                output_results['emd_scores'] = emd_scores
            # other losses
            if class_center == 'MeanWise':
                if self.isda_loss:
                    _, isda_y = torch.max(ytest.view(-1,K), 1)
                # global loss
                ftest = ftest.view(b, n2, n1, -1)
                ftest = ftest.transpose(2, 3)
                ytest = ytest
                ytest = ytest.unsqueeze(3)
                ftest = torch.matmul(ftest, ytest)
                ftest = ftest.view(b * n2, -1, h, w)
                ytest = self.clasifier(ftest)
                if self.rotation_loss:
                    # rotation loss
                    rotate_scores = self.rotate_classifier(ftest) 
                # GlobalFeatMixLoss
                if self.global_feat_mix_loss:
                    global_feat_mix_loss_res = self.GlobalFeatMixLoss(ftest, pids)
                if self.isda_loss:
                    isda_feature = self.isda_avgpool(ftest)
                    isda_feature = isda_feature.view(isda_feature.size(0), -1)
                    isda_loss_res, isda_y_res = self.isda_criterion(isda_feature, self.isda_classifier, isda_y, isda_ratio)
                if self.redundancy_loss:
                    redundancy_loss_res = redundancy_loss_func(ftest)

                ##### output results loss####
                # IMLN losses
                output_results['ytest'] = ytest
                # other losses
                if self.rotation_loss:
                    output_results['rotate_scores'] = rotate_scores
                if self.global_feat_mix_loss:
                    output_results['feat_mix_loss_res'] = feat_mix_loss_res
                if self.isda_loss:
                    output_results['isda_loss_res'] = isda_loss_res
                if self.redundancy_loss:
                    output_results['redundancy_loss_res'] = redundancy_loss_res
            return output_results


class Model(nn.Module):
    def __init__(self, args, backbone_name=None):
        super(Model, self).__init__()
        self.method = args.method
        self.nKnovel = args.nKnovel
        self.nExemplars = args.nExemplars
        self.backbone_name = backbone_name
        if self.backbone_name==None:
            self.backbone = args.backbone
        else:
            self.backbone = self.backbone_name
        self.neck = args.neck
        self.global_classifier_attention = args.global_classifier_attention
        self.palleral_attentions = args.palleral_attentions
        self.num_queries = args.num_queries
        self.num_tSF_novel_queries = args.num_tSF_novel_queries
        self.num_heads = args.num_heads
        self.CECE_mode = args.CECE_mode
        self.class_center = args.class_center
        self.attention = args.attention
        self.attention_pool = args.attention_pool
        self.scale_cls = args.scale_cls
        self.using_power_trans = args.using_power_trans
        self.num_classes = args.num_classes
        self.fix_backbone = args.fix_backbone
        self.fine_tune = args.fine_tune
        self.tSF_plus_mode = args.tSF_plus_mode
        self.tSF_plus_num = args.tSF_plus_num
        self.add_tSF = args.add_tSF
        self.embed_classifier_train = args.embed_classifier_train
        self.embed_classifier_test = args.embed_classifier_test
        self.cluster_embed_by_support = args.cluster_embed_by_support
        self.embed_classifier_weight = args.embed_classifier_weight
        if args.norm_layer == 'bn':
            self.norm_layer = nn.BatchNorm2d
        elif args.norm_layer == 'in':
            self.norm_layer = nn.InstanceNorm2d
        elif args.norm_layer == 'syncbn':
            #self.norm_layer = nn.SyncBatchNorm
            import sys
            sys.path.append("/persons/jinxianglai/FewShotLearning/few-shot_classification/code/CAN")
            from extensions.ops.sync_bn.syncbn import BatchNorm2d
            self.norm_layer = BatchNorm2d
        elif args.norm_layer == 'torchsyncbn':
            self.norm_layer = nn.BatchNorm2d # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        #### backbone ####
        if self.backbone == 'conv4':
            self.base = conv4()
            self.backbone_outsize = [64,19,19]
        elif self.backbone == 'conv4_512':
            self.base = conv4_512()
            self.backbone_outsize = [512,19,19]
        elif self.backbone == 'conv4_512_s':
            self.base = conv4_512_s()
            self.backbone_outsize = [512,19,19]
        elif self.backbone == 'resnet12':
            self.base = resnet12(norm_layer=self.norm_layer) # self.base.nFeat = 512
            self.backbone_outsize = [self.base.nFeat,6,6] # [c,h,w]=[512,6,6] # layer1 stride=2
        elif self.backbone == 'resnet12_avg':
            self.base = resnet12_avg(norm_layer=self.norm_layer) # self.base.nFeat = 512
            self.backbone_outsize = [self.base.nFeat,1,1] # [c,h,w]=[512,6,6] # layer1 stride=2
        elif self.backbone == 'resnet12_gcn':
            self.base = resnet12_gcn(norm_layer=self.norm_layer) # self.base.nFeat = 512
            #self.backbone_outsize = [self.base.nFeat,21,21] # layer1:stride=1;layer2:stride=1
            self.backbone_outsize = [self.base.nFeat,11,11] # layer1 stride=1
            #self.backbone_outsize = [self.base.nFeat,5,5] # layer1 stride=1; pool=2
        elif self.backbone == 'resnet12_gcn_640':
            self.base = resnet12_gcn_640(norm_layer=self.norm_layer) # self.base.nFeat = 640
            self.backbone_outsize = [self.base.nFeat,11,11] # layer1 stride=1
        elif self.backbone == 'resnet12_exp_c640_k3':
            self.base = resnet12_exp_c640_k3(norm_layer=self.norm_layer)
            self.backbone_outsize = [self.base.nFeat,11,11]
        elif self.backbone == 'resnet12_exp_c640_k5':
            self.base = resnet12_exp_c640_k5(norm_layer=self.norm_layer)
            self.backbone_outsize = [self.base.nFeat,11,11]
        elif self.backbone == 'resnet12_exp_c512_k3':
            self.base = resnet12_exp_c512_k3(norm_layer=self.norm_layer)
            self.backbone_outsize = [self.base.nFeat,11,11]
        elif self.backbone == 'resnet12_exp_c768_k3':
            self.base = resnet12_exp_c768_k3(norm_layer=self.norm_layer)
            self.backbone_outsize = [self.base.nFeat,11,11]
        elif self.backbone == 'resnet12_exp_c896_k3':
            self.base = resnet12_exp_c896_k3(norm_layer=self.norm_layer)
            self.backbone_outsize = [self.base.nFeat,11,11]
        elif self.backbone == 'resnet12_exp_c960_k3':
            self.base = resnet12_exp_c960_k3(norm_layer=self.norm_layer)
            self.backbone_outsize = [self.base.nFeat,11,11]
        elif self.backbone == 'resnet12_exp_c1024_k3':
            self.base = resnet12_exp_c1024_k3(norm_layer=self.norm_layer)
            self.backbone_outsize = [self.base.nFeat,11,11]
        elif self.backbone == 'resnet12_gcn_640_tSF':
            self.base = resnet12_gcn_640_tSF(norm_layer=self.norm_layer) # self.base.nFeat = 640
            self.backbone_outsize = [self.base.nFeat,11,11] # layer1 stride=1
        elif self.backbone == 'resnet12_gcn_avg':
            self.base = resnet12_gcn_avg(norm_layer=self.norm_layer) # self.base.nFeat = 512
            self.backbone_outsize = [self.base.nFeat,1,1] # layer1 stride=1
        elif self.backbone == 'ResNet12_BDC':
            self.base = ResNet12_BDC() # self.base.nFeat = 640
            self.backbone_outsize = [self.base.nFeat,10,10] # layer1 stride=1
        elif self.backbone == 'resnet12_cae':
            self.base = resnet12_cae(args, norm_layer=self.norm_layer) # self.base.nFeat = 512
            self.backbone_outsize = [self.base.nFeat,11,11] # layer1 stride=1
            #self.backbone_outsize = [self.base.nFeat,6,6] # [c,h,w]=[512,6,6] # layer1 stride=2
        elif self.backbone == 'rest_small':
            self.base = rest_small() # self.base.nFeat = 512
            #self.backbone_outsize = [self.base.nFeat,11,11] # layer1 stride=1 
            self.backbone_outsize = [self.base.nFeat,6,6]
            #self.backbone_outsize = [self.base.nFeat,7,7]
            #self.backbone_outsize = [self.base.nFeat,1,1]
        elif self.backbone == 'res2net50':
            self.base = res2net50_v1b()
            #self.backbone_outsize = [self.base.nFeat,6,6]
            self.backbone_outsize = [self.base.nFeat,11,11]
        elif self.backbone == 'wrn28_10':
            self.base = wrn28_10(norm_layer=self.norm_layer, num_classes=self.num_classes) # self.base.nFeat = 640
            #self.backbone_outsize = [self.base.nFeat,21,21]
            #self.backbone_outsize = [self.base.nFeat,10,10]
            self.backbone_outsize = [self.base.nFeat,7,7]
            #self.backbone_outsize = [self.base.nFeat,8,8]
            #self.backbone_outsize = [self.base.nFeat,5,5]
        elif self.backbone == 'wrn28_10_16_16_32_64':
            self.base = wrn28_10_16_16_32_64(norm_layer=self.norm_layer, num_classes=self.num_classes)
            self.backbone_outsize = [self.base.nFeat,7,7]
        elif self.backbone == 'wrn28_10_32_32_64_96':
            self.base = wrn28_10_32_32_64_96(norm_layer=self.norm_layer, num_classes=self.num_classes)
            self.backbone_outsize = [self.base.nFeat,7,7]
        elif self.backbone == 'wrn28_10_64_32_64_96':
            self.base = wrn28_10_64_32_64_96(norm_layer=self.norm_layer, num_classes=self.num_classes)
            self.backbone_outsize = [self.base.nFeat,7,7]
        elif self.backbone == 'wrn28_10_cam':
            self.base = wrn28_10_cam(norm_layer=self.norm_layer, num_classes=self.num_classes) # self.base.nFeat = 640
            self.backbone_outsize = [self.base.nFeat,21,21]
        elif self.backbone == 'wrn28_10_gcn':
            self.base = wrn28_10_gcn(norm_layer=self.norm_layer) # self.base.nFeat = 640
            self.backbone_outsize = [self.base.nFeat,10,10] # stride=1, last_pool=2
            #self.backbone_outsize = [self.base.nFeat,11,11] # stride=2, last_pool=None
            #self.backbone_outsize = [self.base.nFeat,11,11] # block4, self.base.nFeat = 720
        elif self.backbone == 'hrnet_w18_small_v1':
            self.base = hrnet_w18_small_v1() # self.base.nFeat = 512
            self.backbone_outsize = [self.base.nFeat,8,8]
            #self.backbone_outsize = [self.base.nFeat,7,7]
            #self.backbone_outsize = [self.base.nFeat,5,5]
        elif self.backbone == 'hrnetv2_w18':
            self.base = hrnetv2_w18() # self.base.nFeat = 512
            self.backbone_outsize = [self.base.nFeat,8,8]
        elif self.backbone == 'densenet121':
            self.base = densenet121()
            self.backbone_outsize = [1024,10,10]
            #self.backbone_outsize = [1024,5,5]
        #self.backbone_outsize[0] = self.backbone_outsize[0]*2

        # neck
        if self.neck == 'Coarse_feat':
            self.class_specific_coarse = ClassSpecific(feature_dim=self.backbone_outsize[0], num_queries=self.num_queries['coarse'], num_heads=self.num_heads, FFN_method='MLP')
        elif self.neck == 'Fine_feat':
            self.class_specific_fine = ClassSpecific(feature_dim=self.backbone_outsize[0], num_queries=self.num_queries['fine'], num_heads=self.num_heads, FFN_method='MLP')
        elif self.neck == 'Task_feat':
            task_num_queries = {}
            task_num_queries['dataset'] = self.num_queries['dataset']
            task_num_queries['query'] = self.nKnovel
            self.task_specific = TaskSpecific(feature_dim=self.backbone_outsize[0], num_queries=task_num_queries, num_heads=self.num_heads, FFN_method='MLP')
        elif self.neck == 'Coarse_Fine':
            self.class_specific_coarse = ClassSpecific(feature_dim=self.backbone_outsize[0], num_queries=self.num_queries['coarse'], num_heads=self.num_heads, FFN_method='None')
            self.class_specific_fine = ClassSpecific(feature_dim=self.backbone_outsize[0], num_queries=self.num_queries['fine'], num_heads=self.num_heads, FFN_method='MLP')
        elif self.neck == 'Coarse_Task':
            self.class_specific_coarse = ClassSpecific(feature_dim=self.backbone_outsize[0], num_queries=self.num_queries['coarse'], num_heads=self.num_heads, FFN_method='None')
            task_num_queries = {}
            task_num_queries['dataset'] = self.num_queries['dataset']
            task_num_queries['query'] = self.nKnovel
            self.task_specific = TaskSpecific(feature_dim=self.backbone_outsize[0], num_queries=task_num_queries, num_heads=self.num_heads, FFN_method='MLP')
        elif self.neck == 'Coarse_Fine_Task':
            self.class_specific_coarse = ClassSpecific(feature_dim=self.backbone_outsize[0], num_queries=self.num_queries['coarse'], num_heads=self.num_heads, FFN_method='None')
            self.class_specific_fine = ClassSpecific(feature_dim=self.backbone_outsize[0], num_queries=self.num_queries['fine'], num_heads=self.num_heads, FFN_method='None')
            task_num_queries = {}
            task_num_queries['dataset'] = self.num_queries['dataset']
            task_num_queries['query'] = self.nKnovel
            self.task_specific = TaskSpecific(feature_dim=self.backbone_outsize[0], num_queries=task_num_queries, num_heads=self.num_heads, FFN_method='MLP')
        elif self.neck == 'tSF_novel':
            self.tSF_novel_encoder = tSF_novel(feature_dim=self.backbone_outsize[0], num_queries=self.num_queries['coarse'], num_novel_queries=self.num_tSF_novel_queries, num_heads=self.num_heads, FFN_method='MLP', layer_num=1)
        elif self.neck == 'tSF_stacker1':
            self.tSF_encoder = tSF_stacker(feature_dim=self.backbone_outsize[0], num_queries=self.num_queries['coarse'], num_heads=self.num_heads, FFN_method='MLP', layer_num=1)
        elif self.neck == 'tSF_stacker2':
            self.tSF_encoder = tSF_stacker(feature_dim=self.backbone_outsize[0], num_queries=self.num_queries['coarse'], num_heads=self.num_heads, FFN_method='MLP', layer_num=2)
        elif self.neck == 'tSF_encoder2':
            self.tSF_encoder = tSF_encoder(feature_dim=self.backbone_outsize[0], num_queries=self.num_queries['coarse'], num_heads=self.num_heads, FFN_method='MLP', layer_num=2)
        elif self.neck == 'SIQ_encoder2':
            self.tSF_encoder = SIQ_encoder(feature_dim=self.backbone_outsize[0], num_queries=self.num_queries['coarse'], num_heads=self.num_heads, FFN_method='MLP', layer_num=2)
        elif self.neck == 'SIQ_encoder3':
            self.tSF_encoder = SIQ_encoder(feature_dim=self.backbone_outsize[0], num_queries=self.num_queries['coarse'], num_heads=self.num_heads, FFN_method='MLP', layer_num=3)
        elif self.neck == 'tSF_T':
            self.tSF_encoder = tSF_T(feature_dim=self.backbone_outsize[0], num_queries=self.num_queries['coarse'], num_heads=self.num_heads, FFN_method='MLP', layer_num=1)
        elif self.neck == 'tSF_T_tSF':
            self.tSF_encoder = tSF_T_tSF(feature_dim=self.backbone_outsize[0], num_queries=self.num_queries['coarse'], num_heads=self.num_heads, FFN_method='MLP', layer_num=1)
        elif self.neck == 'tSF_T_tSF2':
            self.tSF_encoder = tSF_T_tSF(feature_dim=self.backbone_outsize[0], num_queries=self.num_queries['coarse'], num_heads=self.num_heads, FFN_method='MLP', layer_num=2)
        elif self.neck == 'tSF_BDC':
            self.tSF_encoder = tSF_BDC(feature_dims=self.backbone_outsize, num_queries=self.num_queries['coarse'], num_heads=self.num_heads, FFN_method='MLP', layer_num=1)
        elif self.neck == 'MSF':
            self.tSF_encoder = MSF(feature_dim=self.backbone_outsize[0], num_queries=[1,5,10,20], num_heads=self.num_heads, FFN_method='MLP')
        elif self.neck == 'tSF_prototype':
            self.tSF_prototype_encoder = tSF(feature_dim=self.backbone_outsize[0], num_queries=self.num_classes, num_heads=self.num_heads, FFN_method='MLP', base_embedded_prototype_fuse='add')
            # self.tSF_prototype_encoder = tSF(feature_dim=self.backbone_outsize[0], num_queries=self.num_classes, num_heads=self.num_heads, FFN_method='MLP', base_embedded_prototype_fuse='transformer')
            # self.tSF_prototype_encoder = tSF(feature_dim=self.backbone_outsize[0], num_queries=self.num_queries['coarse'], num_heads=self.num_heads, FFN_method='MLP', base_embedded_prototype_fuse='transformer')
        elif self.neck == 'tPF':
            self.tSF_prototype_encoder = tSF_encoder_block(feature_dim=self.backbone_outsize[0], num_heads=self.num_heads, FFN_method='MLP', conv_base_embedded_prototype=True)
        elif self.neck == 'tSF_tPF':
            self.tSF_prototype_encoder = tSF_tPF(feature_dim=self.backbone_outsize[0], num_queries=self.num_queries['coarse'], num_heads=self.num_heads, FFN_method='MLP')
        elif self.neck == 'tSF_plus':
            # tSF_plus_mode: tSF_F, tSF_E, tSF_SP, tSF_BEP, tSF_BEP_SP, tSF_BEP_local, tSF_BEP_global
            if self.tSF_plus_num > 1:
                tSF_plus_layer = tSF_plus(feature_dim=self.backbone_outsize[0], mode=self.tSF_plus_mode, add_tSF=False, num_queries=self.num_queries['coarse'], num_heads=self.num_heads, FFN_method='MLP', base_num_classes=self.num_classes)
                self.tSF_plus_encoders = nn.ModuleList([copy.deepcopy(tSF_plus_layer) for _ in range(self.tSF_plus_num)])
            else:
                self.tSF_plus_encoder = tSF_plus(feature_dim=self.backbone_outsize[0], mode=self.tSF_plus_mode, add_tSF=False, num_queries=self.num_queries['coarse'], num_heads=self.num_heads, FFN_method='MLP', base_num_classes=self.num_classes)
        elif self.neck == 'CECE':
            self.cece_embed = CECE(feature_dim=self.backbone_outsize[0], num_queries=self.num_queries['coarse'], mode=self.CECE_mode)


        # class center
        if self.class_center=='EmbeddingAlignment':
            self.embedding_alignment = EmbeddingAlignment(feature_dim=self.backbone_outsize[0], num_queries=self.backbone_outsize[1]*self.backbone_outsize[2], num_heads=1)

        # fix the layers above (including backbone) in training
        if self.fix_backbone:
            for p in self.parameters(): 
                p.requires_grad=False
        else:
            for p in self.parameters(): 
                p.requires_grad=True

        #### cross attention module and similairty results function ####
        # cross attention module
        self.cross_attention = BuildCrossAttention(args, self.attention, self.backbone_outsize, self.scale_cls, attention_pool=self.attention_pool, norm_layer=self.norm_layer)
        self.cross_attention_none = BuildCrossAttention(args, 'None', self.backbone_outsize, self.scale_cls, attention_pool=self.attention_pool, norm_layer=self.norm_layer)
        if self.attention_pool > 1:
            W = self.backbone_outsize[1]
            F = self.attention_pool
            P = 0
            S = self.attention_pool
            N = (W-F+2*P)//S+1 #N：输出大小; W：输入大小; F：卷积核大小; P：填充值; S：步长
            self.backbone_outsize[1] = N
            self.backbone_outsize[2] = N
        # similairty results function
        self.similarity_res_func = SimilarityResFunc(args, self.method, self.attention, self.backbone_outsize, self.scale_cls, norm_layer=self.norm_layer, embed_classifier_train=self.embed_classifier_train, embed_classifier_test=self.embed_classifier_test, cluster_embed_by_support=self.cluster_embed_by_support, embed_classifier_weight=self.embed_classifier_weight)
        # tSF_prototype
        if ('tSF_prototype' == self.neck) or ('tPF' == self.neck) or ('tSF_tPF' == self.neck) or ('tSF_plus' == self.neck):
            self.attention_clasifier = self.similarity_res_func.clasifier
        # global classifier attention
        if self.global_classifier_attention != 'None':
            self.attention_clasifier = self.similarity_res_func.clasifier

            if self.global_classifier_attention == 'TSCA':
                self.tsca = TSCA(self.backbone_outsize[0], num_heads=1, FFN_method='MLP', mode='TSCA')
            elif self.global_classifier_attention == 'TCA':
                self.tsca = TSCA(self.backbone_outsize[0], num_heads=1, FFN_method='MLP', mode='TCA')
            else:
                self.class_weight_name = 'none'
                #self.class_weight_name = 'mlp'
                #self.class_weight_name = 'conv1'
                if self.class_weight_name == 'mlp':
                    # class weight mlp
                    self.class_weight_FFN = FFN_MLP(self.backbone_outsize[0], d_ffn=self.backbone_outsize[0]*2)
                elif self.class_weight_name == 'conv1':
                    # class weight conv1
                    self.class_weight_conv1 = nn.Conv2d(self.backbone_outsize[0], self.backbone_outsize[0], kernel_size=1)
                # mlp
                self.attention_clasifier_FFN = FFN_MLP(self.backbone_outsize[0], d_ffn=self.backbone_outsize[0]*2)

        # vae loss
        self.vae_loss = args.vae_loss
        if self.vae_loss:
            self.vae_decoder = VAE_decoder(self.backbone_outsize)

        #### visual ####
        self.visual_gcn = False
        if self.visual_gcn:
            descriptor_dim = self.backbone_outsize[0]
            GNN_layers = ['cross'] * 1
            self.cec_relation_map = CECM(descriptor_dim, GNN_layers, mode='Cosine') # mode=['MatMul','Cosine','GCN']

    def f_to_train_test(self, f, ytrain, batch_size, num_train, num_test, class_center='Mean'):
        # class center
        if class_center=='EmbeddingAlignment':
            # EmbeddingAlignment
            f = self.embedding_alignment(f)
            # class mean
            ftrain = f[:batch_size * num_train] # B,c,h,w
            ftrain = ftrain.contiguous().view(batch_size, num_train, -1) 
            ftrain = torch.bmm(ytrain, ftrain)
            ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
            ftrain = ftrain.contiguous().view(batch_size, -1, *f.size()[1:])
            ftest = f[batch_size * num_train:] # B,c,h,w
            ftest = ftest.contiguous().view(batch_size, num_test, *f.size()[1:])
            #print(ftrain.shape) # torch.Size([4, 5, 512, 6, 6])
            #print(ftest.shape) # torch.Size([4, 75, 512, 6, 6])
        else:
            # default: class mean
            ftrain = f[:batch_size * num_train]
            ftrain = ftrain.contiguous().view(batch_size, num_train, -1) 
            ftrain = torch.bmm(ytrain, ftrain)
            ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
            ftrain = ftrain.contiguous().view(batch_size, -1, *f.size()[1:])
            ftest = f[batch_size * num_train:]
            ftest = ftest.contiguous().view(batch_size, num_test, *f.size()[1:])
            #print(ftrain.shape) # torch.Size([4, 5, 512, 6, 6])
            #print(ftest.shape) # torch.Size([4, 75, 512, 6, 6])
        return ftrain, ftest

    def forward(self, xtrain, xtest, ytrain, ytest, ftrain_in=None, ftest_in=None, pids=None, isda_ratio=0,
                manifold_mixup=False, mixup_hidden=True, mixup_alpha=2, lam=0, adapt_metrics=False, 
                using_novel_query=False, finetune_novel_query=False, tSF_BEP_classifier_train=False):
        #print('RelationModel')
        #print(xtrain.shape) # torch.Size([4, 5, 3, 84, 84])
        #print(xtest.shape) # torch.Size([4, 75, 3, 84, 84])
        #print(ytrain.shape) # torch.Size([4, 5, 5]), one_hot label
        #print(ytest.shape) # torch.Size([4, 75, 5]), one_hot label

        if self.backbone == 'resnet12_cae':
            ftrain, ftest = self.base(xtrain, xtest, ytrain, ytest)
            b, num_test, K, c, h, w = ftrain.size()
            # attention results
            attention_results={
                'ftrain': None,
                'ftest': None
            }
            attention_results['ftrain']=ftrain
            attention_results['ftest']=ftest
            # image features
            ftrain = ftrain.mean(1) # b, K, c, h, w
            ftest = ftest.mean(2) # b, num_test, c, h, w
        else:
            if manifold_mixup and self.training:
                xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
                pids = pids.view(-1)
                return self.base(xtest, pids, mixup_hidden=mixup_hidden, mixup_alpha=mixup_alpha, lam=lam)

            if (tSF_BEP_classifier_train == True) and self.training:
                xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
                ftest = self.base(xtest) # num_test, c, h, w
                if self.tSF_plus_num > 1:
                    tSF_BEP_clasifier_res = self.tSF_plus_encoders[0](ftest, tSF_BEP_classifier_train=tSF_BEP_classifier_train)
                else:
                    tSF_BEP_clasifier_res = self.tSF_plus_encoder(ftest, tSF_BEP_classifier_train=tSF_BEP_classifier_train)
                return tSF_BEP_clasifier_res # (num_test, base_num_classes, h, w)

            #### feature ####
            b, num_train = xtrain.size(0), xtrain.size(1)
            num_test = xtest.size(1)
            K = ytrain.size(2)
            ytrain = ytrain.transpose(1, 2)

            xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
            xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
            x = torch.cat((xtrain, xtest), 0)
            if (ftrain_in != None) and (ftest_in != None):
                b, n1, c, h, w = ftrain_in.size()
                b, n2, c, h, w = ftest_in.size()
                ftrain_in_view = ftrain_in.contiguous().view(b*n1, c, h, w)
                ftest_in_view = ftest_in.contiguous().view(b*n2, c, h, w)
                f = torch.cat((ftrain_in_view, ftest_in_view), 0)
            elif ftrain_in != None:
                b, n1, c, h, w = ftrain_in.size()
                ftrain_in_view = ftrain_in.contiguous().view(b*n1, c, h, w)
                ftest_ = self.base(xtest)
                f = torch.cat((ftrain_in_view, ftest_), 0)
            else:
                f = self.base(x)
            B, c, h, w = f.size()
            #print(f.shape) # resnet12: torch.Size([320, 512, 6, 6])

            # neck: ClassSpecific
            if 'Coarse_Fine' in self.neck:
                f, softmax_qk_coarse = self.class_specific_coarse(f)
                f, softmax_qk_fine = self.class_specific_fine(f)
            elif 'Coarse' in self.neck:
                f, softmax_qk_coarse = self.class_specific_coarse(f)
            elif 'Fine' in self.neck:
                f, softmax_qk_fine = self.class_specific_fine(f)
            elif 'tSF_novel' in self.neck:
                f = self.tSF_novel_encoder(f, using_novel_query=using_novel_query)
            elif ('tSF_prototype' == self.neck) or ('tPF' == self.neck) or ('tSF_tPF' == self.neck):
                class_weight = self.attention_clasifier.weight.data # (self.num_classes, c, 1, 1)
                class_weight = class_weight.squeeze(2).squeeze(2) # (self.num_classes, c)
                f = self.tSF_prototype_encoder(f, base_embedded_prototype=class_weight)
            elif 'tSF_plus' == self.neck:
                # base embedded-prototype
                class_weight = self.attention_clasifier.weight.data # (self.num_classes, c, 1, 1)
                class_weight = class_weight.squeeze(2).squeeze(2) # (self.num_classes, c)
                # support_prototype
                support_prototype, _ = self.f_to_train_test(f, ytrain, b, num_train, num_test, class_center='Mean')
                support_prototype = support_prototype.mean(3)
                support_prototype = support_prototype.mean(3) # (b, K, c)
                # data_configs
                data_configs = {}
                data_configs['batch_size'] = b
                data_configs['num_train'] = num_train
                data_configs['num_test'] = num_test
                # attention
                if self.tSF_plus_num > 1:
                    for tSF_plus_layer in self.tSF_plus_encoders:
                        f = tSF_plus_layer(f, base_embedded_prototype=class_weight, support_prototype=support_prototype, data_configs=data_configs, tSF_BEP_classifier_train=tSF_BEP_classifier_train)
                        if self.tSF_plus_mode == 'tSF_E_Metric':
                            query_embed = tSF_plus_layer.query_embed # (num_queries, c)
                else:
                    f = self.tSF_plus_encoder(f, base_embedded_prototype=class_weight, support_prototype=support_prototype, data_configs=data_configs, tSF_BEP_classifier_train=tSF_BEP_classifier_train)
                    if self.tSF_plus_mode == 'tSF_E_Metric':
                        query_embed = self.tSF_plus_encoder.query_embed # (num_queries, c)
            elif ('tSF' in self.neck) or ('SIQ' in self.neck):
                f = self.tSF_encoder(f)
            elif 'CECE' in self.neck:
                f = self.cece_embed(f)

            # global classifier attention
            if self.global_classifier_attention != 'None':
                if (self.global_classifier_attention == 'TSCA') or (self.global_classifier_attention == 'TCA'):
                    class_weight = self.attention_clasifier.weight.data # (self.num_classes, c, 1, 1)
                    class_weight = class_weight.squeeze(2).squeeze(2) # (self.num_classes, c)
                    f_attention = self.tsca(f, base_class_weight=class_weight)
                else:
                    # dense classifier
                    dense_clasifier_res = self.attention_clasifier(f) # (B, self.num_classes, h, w)
                    # dense res
                    dense_clasifier_res = dense_clasifier_res.view(B, self.num_classes, h*w)
                    dense_clasifier_res = F.softmax(dense_clasifier_res, dim=1)
                    # mean res
                    mean_clasifier_res = torch.mean(dense_clasifier_res, dim=2)
                    mean_clasifier_res = mean_clasifier_res.view(B, self.num_classes)
                    if self.global_classifier_attention == 'ClassWeightAttention':
                        # select max similarity map
                        _, preds = torch.max(mean_clasifier_res, 1)
                        # class weight
                        class_weight = self.attention_clasifier.weight.data
                        L_norm = torch.norm(class_weight, p=2, dim=1).unsqueeze(1).expand_as(class_weight)
                        class_weight = class_weight.div(L_norm + 0.00001)
                        # class_weight = (self.num_classes, c, 1, 1)
                        # select class weight
                        class_weight_attention = class_weight[preds] # (B, c, 1, 1)
                    elif self.global_classifier_attention == 'ThreshClassAttention':
                        # class weight
                        class_weight = self.attention_clasifier.weight.data
                        # all class weight
                        # class_weight = (self.num_classes, c, 1, 1)
                        thresh = 0.5
                        res_thresh = mean_clasifier_res * (mean_clasifier_res > thresh)
                        class_weight_attention = torch.matmul(res_thresh.unsqueeze(1), class_weight.squeeze(2).squeeze(2)).squeeze(1) # (B, c)
                        class_weight_attention = class_weight_attention.unsqueeze(2).unsqueeze(3) # (B, c, 1, 1)
                        # norm
                        attention_norm = torch.norm(class_weight_attention, p=2, dim=1).unsqueeze(1).expand_as(class_weight_attention)
                        class_weight_attention = class_weight_attention.div(attention_norm + 0.00001)

                    # class weight transform
                    if self.class_weight_name == 'mlp':
                        B, c, h_a, w_a = class_weight_attention.size()
                        # class weight mlp
                        class_weight_attention = class_weight_attention.contiguous().view(B, c, h_a*w_a).transpose(1, 2)
                        class_weight_attention = self.class_weight_FFN(class_weight_attention)
                        class_weight_attention = class_weight_attention.transpose(1, 2).contiguous().view(B, c, h_a, w_a)
                    elif self.class_weight_name == 'conv1':
                        # class weight conv1
                        class_weight_attention = self.class_weight_conv1(class_weight_attention)
                    # attention
                    f_attention = f * (class_weight_attention + 1) # multiply
                    # f_attention = f + (class_weight_attention + 1) # add
                    # mlp
                    f_attention = f_attention.contiguous().view(B, c, h*w).transpose(1, 2) # (B, h*w, c)
                    f_attention = self.attention_clasifier_FFN(f_attention)
                    f_attention = f_attention.transpose(1, 2).contiguous().view(B, c, h, w) # (B, c, h, w)
                if self.palleral_attentions == False:
                    f = f_attention


            if not self.training and (self.backbone_name == None) and (finetune_novel_query == False): 
                # Power transform
                if self.using_power_trans and self.nExemplars > 1 and (ftrain_in == None) and (ftest_in == None):
                    beta = 0.5 # defauit=0.5
                    f = torch.pow(f+1e-6, beta)
                    if self.palleral_attentions and (self.global_classifier_attention != 'None'):
                        f_attention = torch.pow(f_attention+1e-6, beta)

            #### transfer f to ftrain and ftest ####
            if (ftrain_in != None) and (ftest_in != None):
                b, n1, c, h, w = ftrain_in.size()
                ftrain = f[:b*n1]
                ftest = f[b*n1:]
                ftrain = ftrain.contiguous().view(b, -1, *f.size()[1:])
                ftest = ftest.contiguous().view(b, -1, *f.size()[1:])
            elif ftrain_in != None:
                b, n1, c, h, w = ftrain_in.size()
                ftrain = f[:b*n1]
                ftest = f[b*n1:]
                ftrain = ftrain.contiguous().view(b, -1, *f.size()[1:])
                ftest = ftest.contiguous().view(b, -1, *f.size()[1:])
            else:
                ftrain, ftest = self.f_to_train_test(f, ytrain, b, num_train, num_test, class_center=self.class_center)
            if self.palleral_attentions and (self.global_classifier_attention != 'None'):
                # ftrain_attention, ftest_attention = self.f_to_train_test(f_attention, ytrain, b, num_train, num_test, class_center=self.class_center)
                if (ftrain_in != None) and (ftest_in != None):
                    b, n1, c, h, w = ftrain_in.size()
                    ftrain_attention = f_attention[:b*n1]
                    ftest_attention = f_attention[b*n1:]
                    ftrain_attention = ftrain_attention.contiguous().view(b, -1, *f_attention.size()[1:])
                    ftest_attention = ftest_attention.contiguous().view(b, -1, *f_attention.size()[1:])
                elif ftrain_in != None:
                    b, n1, c, h, w = ftrain_in.size()
                    ftrain_attention = f_attention[:b*n1]
                    ftest_attention = f_attention[b*n1:]
                    ftrain_attention = ftrain_attention.contiguous().view(b, -1, *f_attention.size()[1:])
                    ftest_attention = ftest_attention.contiguous().view(b, -1, *f_attention.size()[1:])
                else:
                    ftrain_attention, ftest_attention = self.f_to_train_test(f_attention, ytrain, b, num_train, num_test, class_center=self.class_center)
            #print(ftrain.shape) # torch.Size([4, 5, 512, 6, 6])
            #print(ftest.shape) # torch.Size([4, 75, 512, 6, 6])

            # neck: TaskSpecific
            if 'Task' in self.neck:
                ftest = self.task_specific(ftrain, ftest)

            ##### MeanWise results ####
            ## cross attention module ##
            attention_results = self.cross_attention(ftrain, ftest)

            if self.palleral_attentions and (self.global_classifier_attention != 'None'):
                classifier_attention_results = self.cross_attention_none(ftrain_attention, ftest_attention)
                attention_results['ftrain'] = attention_results['ftrain'] + classifier_attention_results['ftrain']
                attention_results['ftest'] = attention_results['ftest'] + classifier_attention_results['ftest']

        ## testing and loss ##
        if ('tSF_plus' == self.neck) and (self.tSF_plus_mode == 'tSF_E_Metric'):
            output_results = self.similarity_res_func(ytest, attention_results, backbone_name=self.backbone_name, pids=pids, isda_ratio=isda_ratio, adapt_metrics=adapt_metrics, finetune_novel_query=finetune_novel_query, query_embed=query_embed)
        else:
            output_results = self.similarity_res_func(ytest, attention_results, backbone_name=self.backbone_name, pids=pids, isda_ratio=isda_ratio, adapt_metrics=adapt_metrics, finetune_novel_query=finetune_novel_query)

        # visual the feature
        ftest_visual = None
        if self.visual_gcn:
            #ftest_visual = ftest # (b, n2, c, h, w)
            '''# softmax_qk
            #softmax_qk_coarse # (B, h*w, num_queries), B=b*(n1+n2)
            ftest_qk = softmax_qk_coarse[b * num_train:]
            num_queries = ftest_qk.size(2)
            ftest_qk = ftest_qk.contiguous().view(-1, num_queries) # (B*h*w, num_queries)
            ftest_qk = ftest_qk.transpose(0, 1) # (num_queries, B*h*w)
            ftest_qk_i = ftest_qk[0] # (B*h*w)
            ftest_qk_i = ftest_qk_i.contiguous().view(b, num_test, h, w) # (b, n2, h, w)
            ftest_qk_i = ftest_qk_i.unsqueeze(2) # (b, n2, 1, h, w)
            ftest_visual = ftest_qk_i
            '''
            '''
            # ftest after gcn attention
            ftest_visual = attention_results['ftest']
            b, n2, n1, c, h, w = ftest_visual.size()
            ftest_visual = ftest_visual.contiguous().view(b, n2, n1, -1)
            ftest_visual = ftest_visual.transpose(2, 3) 
            ytest = ytest.unsqueeze(3)
            ftest_visual = torch.matmul(ftest_visual, ytest)
            ftest_visual = ftest_visual.view(b, n2, c, h, w)
            #'''
            '''# relation map for Clustered-patch activation map
            ftrain_visual = attention_results['ftrain']
            ftest_visual = attention_results['ftest']
            b, n2, n1, c, h, w = ftrain_visual.size()
            ftrain_ = ftrain_visual.contiguous().view(b*n2*n1, c, h*w)
            ftest_ = ftest_visual.contiguous().view(b*n2*n1, c, h*w)
            _, _, relation_map_ftrain, relation_map_ftest = self.cec_relation_map(ftrain_, ftest_)
            cls_scores = relation_map_ftest.contiguous().view(b, n2, n1, h, w)
            #'''      
            #'''# similarity map for calss activation map
            ftrain_visual = attention_results['ftrain']
            ftest_visual = attention_results['ftest']
            b, n2, n1, c, h, w = ftrain_visual.size()
            ftrain_visual = ftrain_visual.mean(4)
            ftrain_visual = ftrain_visual.mean(4)
            ftest_norm = F.normalize(ftest_visual, p=2, dim=3, eps=1e-12)
            ftrain_norm = F.normalize(ftrain_visual, p=2, dim=3, eps=1e-12)
            ftrain_norm = ftrain_norm.unsqueeze(4)
            ftrain_norm = ftrain_norm.unsqueeze(5)
            cls_scores = self.scale_cls * torch.sum(ftest_norm * ftrain_norm, dim=3)
            #'''
            #'''# output visual ftest
            ftest_visual = cls_scores.contiguous().view(b, n2, n1, -1)
            ftest_visual = ftest_visual.transpose(2, 3) 
            ytest = ytest.unsqueeze(3)
            ftest_visual = torch.matmul(ftest_visual, ytest)
            ftest_visual = ftest_visual.view(b, n2, -1, h, w)
            #'''
        if self.vae_loss and self.training:
            ftest_vae = f[b * num_train:]
            recon_x, mu, logvar = self.vae_decoder(ftest_vae)
            #print(recon_x.shape) # [3,24,24]
            vae_loss_res = vae_loss_func(recon_x, xtest, mu, logvar)
            output_results['vae_loss_res'] = vae_loss_res
        return output_results, ftrain, ftest, ftest_visual

    def helper(self, ftrain, ftest, ytrain):
        b, n, c, h, w = ftrain.size()
        k = ytrain.size(2)

        ytrain_transposed = ytrain.transpose(1, 2)
        ftrain = torch.bmm(ytrain_transposed, ftrain.view(b, n, -1))
        ftrain = ftrain.div(ytrain_transposed.sum(dim=2, keepdim=True).expand_as(ftrain))
        ftrain = ftrain.view(b, -1, c, h, w)

        ftrain, ftest = self.cam(ftrain, ftest)
        ftrain = ftrain.mean(-1).mean(-1)
        ftest = ftest.mean(-1).mean(-1)

        ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
        ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
        scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1)
        return scores

    def test_transductive(self, xtrain, xtest, ytrain, ytest):
        iter_num_prob = self.iter_num_prob
        b, num_train = xtrain.size(0), xtrain.size(1)
        num_test = xtest.size(1)
        K = ytrain.size(2)

        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))
        x = torch.cat((xtrain, xtest), 0)
        f = self.base(x)

        ftrain = f[: b*num_train].view(b, num_train, *f.size()[1:])
        ftest = f[b*num_train:].view(b, num_test, *f.size()[1:])
        cls_scores = self.helper(ftrain, ftest, ytrain)

        num_images_per_iter = int(num_test * iter_num_prob)
        num_iter = num_test // num_images_per_iter

        for i in range(num_iter):
            max_scores, preds = torch.max(cls_scores, 2)
            chose_index = torch.argsort(max_scores.view(-1), descending=True)
            chose_index = chose_index[: num_images_per_iter * (i + 1)]

            ftest_iter = ftest[0, chose_index].unsqueeze(0)
            preds_iter = preds[0, chose_index].unsqueeze(0)
            preds_iter = one_hot(preds_iter).cuda()

            ftrain_iter = torch.cat((ftrain, ftest_iter), 1)
            ytrain_iter = torch.cat((ytrain, preds_iter), 1)
            cls_scores = self.helper(ftrain_iter, ftest, ytrain_iter)

        return cls_scores

if __name__ == '__main__':
    torch.manual_seed(0)

    net = Model()
    net.eval()

    x1 = torch.rand(1, 5, 3, 84, 84)
    x2 = torch.rand(1, 75, 3, 84, 84)
    y1 = torch.rand(1, 5, 5)
    y2 = torch.rand(1, 75, 5)

    y1 = net.test_transductive(x1, x2, y1, y2)
    print(y1.size())
