from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .bdc import BDC

# self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim)

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

def mutual_nearest_attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    prob2 = torch.nn.functional.softmax(scores, dim=-2)
    prob = prob * prob2
    prob = torch.nn.functional.softmax(prob / 0.025, dim=-1)
    # max
    prob = (prob == prob.max(dim=-1, keepdim=True)[0]).to(dtype=torch.float32)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int, with_W=True, mutual_nearest_att=False):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.with_W = with_W
        self.mutual_nearest_att = mutual_nearest_att
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

        if self.mutual_nearest_att:
            x, softmax_qk = mutual_nearest_attention(query, key, value)
        else:
            x, softmax_qk = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1)), softmax_qk

##############################################

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

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """
    # usage: self.attention = ScaledDotProductAttention(temperature=np.power(feat_dim, 0.5))
    def __init__(self, temperature, attn_dropout=0.0, mutual_nearest_att=False):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.mutual_nearest_att = mutual_nearest_att

    def forward(self, q, k, v):
        if self.mutual_nearest_att:
            attn = torch.bmm(q, k.transpose(1, 2))
            attn = attn / self.temperature
            log_attn = F.log_softmax(attn, 2)
            # mutual_nearest_att
            attn1 = torch.nn.functional.softmax(attn, dim=-1)
            attn2 = torch.nn.functional.softmax(attn, dim=-2)
            attn = attn1 * attn2
            attn = torch.nn.functional.softmax(attn / 0.025, dim=-1)
            # max
            attn = (attn == attn.max(dim=-1, keepdim=True)[0]).to(dtype=torch.float32)
            # out
            # attn = self.dropout(attn)
            output = torch.bmm(attn, v)
        else:
            attn = torch.bmm(q, k.transpose(1, 2))
            attn = attn / self.temperature
            log_attn = F.log_softmax(attn, 2)
            attn = self.softmax(attn)
            attn = self.dropout(attn)
            output = torch.bmm(attn, v)
        return output, attn, log_attn


##############################################
class EmbeddingAlignment(nn.Module):
    def __init__(self, feature_dim, num_queries=100, num_heads=1):
        super(EmbeddingAlignment, self).__init__()
        self.feature_dim = feature_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        # query embedding
        self.query_embed = nn.Embedding(self.num_queries, self.feature_dim)
        # attention
        self.with_W = False
        self.mutual_nearest_att = True
        if self.num_heads > 1:
            self.attention = MultiHeadedAttention(self.num_heads, self.feature_dim, with_W=self.with_W, mutual_nearest_att=self.mutual_nearest_att) # (B, C, N)
        else:
            # with W
            if self.with_W:
                self.merge = nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=1, bias=False)
                self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
            self.attention = ScaledDotProductAttention(temperature=np.power(self.feature_dim, 0.5), mutual_nearest_att=self.mutual_nearest_att) # (B, N, C)

    def forward(self, src):
        """
        Shape: 
        - src.size = B, c, h, w. B=batch size
        """
        B, c, h, w = src.size()
        assert c == self.feature_dim
        src = src.contiguous().view(B, c, h*w).transpose(1, 2) # (B, h*w, c)
        # feature interaction
        query_embed_weight = self.query_embed.weight.unsqueeze(0).repeat(B,1,1) # (B, num_queries, c)
        q = query_embed_weight # (B, num_queries, c)
        k = src # (B, h*w, c)
        v = src # (B, h*w, c)
        if self.num_heads > 1:
            output, softmax_qk = self.attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)) # (B, c, h*w)
            output = output.transpose(1, 2).contiguous().view(B, c, h, w) # (B, c, h, w)
        else:
            # with W
            if self.with_W:
                q, k, v = [l(x) for l, x in zip(self.proj, (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)))]
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

            output, _, softmax_qk = self.attention(q, k, v) # (B, h*w, c)
            output = output.transpose(1, 2).contiguous().view(B, c, h, w) # (B, c, h, w)
        # return output, softmax_qk
        return output
