from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "leaky_relu":
        return F.leaky_relu
    raise RuntimeError(F"activation should be relu/gelu/glu/leaky_relu, not {activation}.")


class FFN_MLP(nn.Module):
    def __init__(self, feature_dim, d_ffn=1024, dropout=0.1, activation="relu"):
        super(FFN_MLP, self).__init__()
        self.linear1 = nn.Linear(feature_dim, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, feature_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(feature_dim)

    def forward(self, src):
        src2 = self.linear2(self.dropout3(self.activation(self.linear1(src))))
        src = src + self.dropout4(src2)
        src = self.norm3(src)
        return src

def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, _ = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        #self.v_embed = nn.Embedding(11*11, feature_dim) # resnet12_gcn
        #self.v_embed = nn.Embedding(7*7, feature_dim) # wrn28_10
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        B, C, N = x.size()
        message = self.attn(x, source, source)
        #v = self.v_embed.weight.transpose(0, 1).unsqueeze(0).repeat(B,1,1) # (B, C, N)
        #message = self.attn(x, source, v)
        return self.mlp(torch.cat([x, message], dim=1))

class PatchCluster(nn.Module):
    def __init__(self, feature_dim: int, mode='GCN'):
        super(PatchCluster, self).__init__()
        self.mode = mode # mode=['MatMul','Cosine','GCN']
        if self.mode == 'GCN':
            self.gcn_weight = nn.Conv1d(feature_dim, feature_dim, 1)
            self.relu = nn.LeakyReLU(0.2)
        elif self.mode == 'Transformer':
            self.transformer_layer = AttentionalPropagation(feature_dim, 4)

    def forward(self, x, source):
        # {query, key, value} = {x, source, source}
        B, N, C = x.size()
        if self.mode == 'MatMul':
            correlation = torch.matmul(x, source.transpose(1, 2)) # (B, N, N)
        elif self.mode == 'Cosine' or self.mode == 'GCN':
            x_norm = torch.norm(x, p=2, dim=-1).unsqueeze(2).expand_as(x)
            x_norm = x.div(x_norm + 0.00001)
            source_norm = torch.norm(source, p=2, dim=-1).unsqueeze(2).expand_as(source)
            source_norm = source.div(source_norm + 0.00001)
            correlation = torch.matmul(x_norm, source_norm.transpose(1, 2)) # (B, N, N)

        # clustered_patch
        if self.mode == 'Transformer':
            clustered_patch = self.transformer_layer(x.transpose(1, 2), source.transpose(1, 2))
            clustered_patch = clustered_patch.transpose(1, 2) # (B, N, C)
        else:
            clustered_patch = torch.matmul(F.softmax(correlation, dim=-1), source) # (B, N, C)
            if self.mode == 'GCN':
                clustered_patch = self.gcn_weight(clustered_patch.transpose(1, 2))
                clustered_patch = self.relu(clustered_patch)
                clustered_patch = clustered_patch.transpose(1, 2)
        return clustered_patch


class CECM(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, mode='Transformer'):
        super(CECM, self).__init__()
        self.layers = nn.ModuleList([
            PatchCluster(feature_dim, mode=mode)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, support, query):
        """
        Shape: 
        - support.size = B, C, N. B=batch size, C=channel, N=h*w
        """
        support = support.transpose(1, 2) # (B, N, C)
        query = query.transpose(1, 2) # (B, N, C)
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = query, support
            else:  # if name == 'self':
                src0, src1 = support, query
            clustered_patch0, clustered_patch1 = layer(support, src0), layer(query, src1) # QCp
            #clustered_patch1, clustered_patch0 = layer(support, src0), layer(query, src1) # PCp

            # Element connection
            # support
            desc0_norm = torch.norm(support, p=2, dim=-1).unsqueeze(2).expand_as(support)
            desc0_norm = support.div(desc0_norm + 0.00001)
            clustered_patch0_norm = torch.norm(clustered_patch0, p=2, dim=-1).unsqueeze(2).expand_as(clustered_patch0)
            clustered_patch0_norm = clustered_patch0.div(clustered_patch0_norm + 0.00001)
            cos_scores0 = torch.sum(desc0_norm * clustered_patch0_norm, dim=-1)
            relation_map0 = cos_scores0
            #relation_map0 = F.softmax(relation_map0 / 0.025, dim=-1)
            cos_scores0 = F.softmax(cos_scores0 / 0.025, dim=-1) + 1 # (B, N)
            cos_scores0 = cos_scores0.unsqueeze(1) # (B, 1, N)
            support = support.transpose(1, 2) * cos_scores0 # (B, C, N)

            # query
            desc1_norm = torch.norm(query, p=2, dim=-1).unsqueeze(2).expand_as(query)
            desc1_norm = query.div(desc1_norm + 0.00001)
            clustered_patch1_norm = torch.norm(clustered_patch1, p=2, dim=-1).unsqueeze(2).expand_as(clustered_patch1)
            clustered_patch1_norm = clustered_patch1.div(clustered_patch1_norm + 0.00001)
            cos_scores1 = torch.sum(desc1_norm * clustered_patch1_norm, dim=-1)
            relation_map1 = cos_scores1
            #relation_map1 = F.softmax(relation_map1 / 0.025, dim=-1)
            cos_scores1 = F.softmax(cos_scores1 / 0.025, dim=-1) + 1 # (B, N)
            cos_scores1 = cos_scores1.unsqueeze(1) # (B, 1, N)
            query = query.transpose(1, 2) * cos_scores1 # (B, C, N)

            ##support = support + clustered_patch0.transpose(1, 2) # (B, C, N)
            ##query = query + clustered_patch1.transpose(1, 2) # (B, C, N)

        return support, query, relation_map0, relation_map1

class CECD(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, mode='Cosine', scale_cls=7):
        super(CECD, self).__init__()
        self.layers = nn.ModuleList([
            PatchCluster(feature_dim, mode=mode)
            for _ in range(len(layer_names))])
        self.names = layer_names
        self.scale_cls = scale_cls

    def forward(self, support, query):
        """
        Shape: 
        - support.size = B, C, N. B=batch size, C=channel, N=h*w
        """
        support = support.transpose(1, 2) # (B, N, C)
        query = query.transpose(1, 2) # (B, N, C)
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = query, support
            else:  # if name == 'self':
                src0, src1 = support, query

            # similarity
            ''' support
            clustered_patch0 = layer(support, src0) # QCp
            desc0_norm = torch.norm(support, p=2, dim=-1).unsqueeze(2).expand_as(support)
            desc0_norm = support.div(desc0_norm + 0.00001)
            clustered_patch0_norm = torch.norm(clustered_patch0, p=2, dim=-1).unsqueeze(2).expand_as(clustered_patch0)
            clustered_patch0_norm = clustered_patch0.div(clustered_patch0_norm + 0.00001)
            cos_scores0 = self.scale_cls * torch.sum(desc0_norm * clustered_patch0_norm, dim=-1) # (B, N)
            '''
            # query
            clustered_patch1 = layer(query, src1) # QCp
            desc1_norm = torch.norm(query, p=2, dim=-1).unsqueeze(2).expand_as(query)
            desc1_norm = query.div(desc1_norm + 0.00001)
            clustered_patch1_norm = torch.norm(clustered_patch1, p=2, dim=-1).unsqueeze(2).expand_as(clustered_patch1)
            clustered_patch1_norm = clustered_patch1.div(clustered_patch1_norm + 0.00001)
            cos_scores1 = self.scale_cls * torch.sum(desc1_norm * clustered_patch1_norm, dim=-1) # (B, N)

        return cos_scores1


# embedding block as a neck inserted after the backbone
class CECE(nn.Module):
    def __init__(self, feature_dim, num_queries=5, mode='Transformer'):
        super(CECE, self).__init__()
        self.feature_dim = feature_dim
        self.num_queries = num_queries
        # query embedding
        self.query_embed = nn.Embedding(self.num_queries, self.feature_dim)
        # PatchCluster
        self.patch_cluster = PatchCluster(self.feature_dim, mode=mode)
        # FFN
        self.FFN = FFN_MLP(self.feature_dim, d_ffn=self.feature_dim*2)

    def forward(self, src):
        """
        Shape: 
        - src.size = B, c, h, w. B=batch size
        """
        B, c, h, w = src.size()
        # PatchCluster
        src = src.contiguous().view(B, c, h*w)
        src = src.transpose(1, 2) # (B, h*w, C)
        query_embed_weight = self.query_embed.weight.unsqueeze(0).repeat(B,1,1) # (B, num_queries, c)
        clustered_patch0 = self.patch_cluster(src, query_embed_weight) # {query, key, value} = {src, query_embed_weight, query_embed_weight}
        # Element connection
        desc0_norm = torch.norm(src, p=2, dim=-1).unsqueeze(2).expand_as(src)
        desc0_norm = src.div(desc0_norm + 0.00001)
        clustered_patch0_norm = torch.norm(clustered_patch0, p=2, dim=-1).unsqueeze(2).expand_as(clustered_patch0)
        clustered_patch0_norm = clustered_patch0.div(clustered_patch0_norm + 0.00001)
        cos_scores0 = torch.sum(desc0_norm * clustered_patch0_norm, dim=-1)
        cos_scores0 = cos_scores0.unsqueeze(1) # (B, 1, h*w)
        src = src.transpose(1, 2) * (cos_scores0 + 1) # (B, C, h*w)
        # FFN
        src = src.transpose(1, 2) # (B, h*w, c)
        src = self.FFN(src)  # (B, h*w, c)
        # output
        src = src.transpose(1, 2).contiguous().view(B, c, h, w) # (B, c, h, w)
        return src
