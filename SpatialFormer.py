from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


## multi-head attention ##
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
    def __init__(self, num_heads: int, d_model: int, with_W=True):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.with_W = with_W
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        # B, C, N = query.size()
        batch_dim = query.size(0)    
        if self.with_W:
            # with W
            query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                                for l, x in zip(self.proj, (query, key, value))]
        else:
            # without W
            query = query.view(batch_dim, self.dim, self.num_heads, -1)
            key = key.view(batch_dim, self.dim, self.num_heads, -1)
            value = value.view(batch_dim, self.dim, self.num_heads, -1)

        x, softmax_qk = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1)), softmax_qk

## single-head attention ##
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

class FFN_MLP_one(nn.Module):
    def __init__(self, feature_dim, scale=1.0):
        super(FFN_MLP_one, self).__init__()
        self.linear1 = nn.Linear(feature_dim, feature_dim)

    def forward(self, src):
        src = scale * self.linear1(src) + src # (B, h*w, c)
        return src

##############################################

# SFTA: SpatialFormer Target Attention
# TTA: Transformer Target Attention
class SFTA(nn.Module):
    def __init__(self, feature_dim, num_heads=1, FFN_method='MLP', mode='SFTA', softmax_score=False):
        super(SFTA, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.FFN_method = FFN_method
        self.mode = mode
        self.softmax_score = softmax_score
        # attention
        self.with_W = True
        self.attention = MultiHeadedAttention(self.num_heads, self.feature_dim, with_W=self.with_W) # (B, C, N)
        if self.FFN_method == 'MLP':
            self.FFN = FFN_MLP(self.feature_dim, d_ffn=self.feature_dim*2)
        elif self.FFN_method == 'MLP_one':
            self.FFN = FFN_MLP_one(self.feature_dim, scale=0.01)

    def forward(self, feature, base_class_weight=None, attention_feature=None):
        """
        input Shape: 
        - feature.size = B, c, h, w. B=batch size
        - base_class_weight.size = base_class_num, c.
        - attention_feature.size = B, c, h, w. B=batch size
        output Shape: 
        - output.size = B, c, h, w.
        """
        if (base_class_weight == None) and (attention_feature == None):
            return feature
        B, c, h, w = feature.size()
        feature = feature.contiguous().view(B, c, h*w).transpose(1, 2) # (B, h*w, c)
        # feature interaction
        q = feature # (B, h*w, c)
        if base_class_weight != None:
            k = v = base_class_weight.repeat(B,1,1) # (B, base_class_num, c)
        elif attention_feature != None:
            k = v = attention_feature.contiguous().view(B, c, h*w).transpose(1, 2) # (B, h*w, c)
        aligned_feature, softmax_qk = self.attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)) # (B, c, h*w)

        if self.mode == 'SFTA':
            # Element cosine similarity
            # aligned_feature norm
            aligned_feature = aligned_feature.transpose(1, 2) # (B, h*w, c)
            aligned_feature_norm = torch.norm(aligned_feature, p=2, dim=-1).unsqueeze(2).expand_as(aligned_feature)
            aligned_feature_norm = aligned_feature.div(aligned_feature_norm + 0.00001)
            # feature norm
            feature_norm = torch.norm(feature, p=2, dim=-1).unsqueeze(2).expand_as(feature)
            feature_norm = feature.div(feature_norm + 0.00001)
            # similarity
            cos_scores = torch.sum(aligned_feature_norm * feature_norm, dim=-1)
            if self.softmax_score:
                cos_scores = F.softmax(cos_scores / 0.025, dim=-1) # (B, h*w)
            # spatial attention
            cos_scores = cos_scores.unsqueeze(1) # (B, 1, h*w)
            output = feature.transpose(1, 2) * (cos_scores + 1) # (B, c, h*w)
            output = output.transpose(1, 2) # (B, h*w, c)
        elif self.mode == 'TTA':
            output = aligned_feature.transpose(1, 2) + feature  # (B, h*w, c)

        # FFN_MLP
        if self.FFN_method == 'MLP':
            output = self.FFN(output)  # (B, h*w, c)
        elif self.FFN_method == 'MLP_one':
            output = self.FFN(output)  # (B, h*w, c)
        output = output.transpose(1, 2).contiguous().view(B, c, h, w) # (B, c, h, w)
        return output


# SFSA: SpatialFormer Semantic Attention
# TSA: Transformer Semantic Attention
class SFSA(nn.Module):
    def __init__(self, feature_dim, num_heads=1, FFN_method='MLP', mode='SFSA'):
        super(SFSA, self).__init__()
        if mode == 'SFSA':
            self.sfta = SFTA(feature_dim, num_heads=num_heads, FFN_method=FFN_method, mode='SFTA', softmax_score=True)
        elif mode == 'TSA':
            self.sfta = SFTA(feature_dim, num_heads=num_heads, FFN_method=FFN_method, mode='TTA')

    def forward(self, support, query):
        """
        input Shape: 
        - support.size = B, c, h, w. B=batch size
        - query.size = B, c, h, w. B=batch size
        output Shape: 
        - support.size = B, c, h, w. B=batch size
        - query.size = B, c, h, w. B=batch size
        """
        support = self.sfta(support, attention_feature=query)
        query = self.sfta(query, attention_feature=support)
        return support, query

