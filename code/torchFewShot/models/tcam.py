from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


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
    prob2 = torch.nn.functional.softmax(scores, dim=-2)
    prob = prob * prob2
    prob = torch.nn.functional.softmax(prob / 0.025, dim=-1)
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
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))


class TCAM(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 4)
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1):
        """
        Shape: 
        - desc0.size = B, C, N. B=batch size, C=channel, N=h*w
        """
        for layer, name in zip(self.layers, self.names):
            if name == 'cross':
                src0, src1 = desc1, desc0
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)

            # spatial-attention
            delta0, delta1 = (desc0 + delta0), (desc1 + delta1)
            # delta0
            delta0_spatial_attention = delta0.mean(1) # mean
            delta0_spatial_attention = F.softmax(delta0_spatial_attention / 0.025, dim=-1)
            delta0_spatial_attention = delta0_spatial_attention.unsqueeze(1)# (B, 1, N)
            delta0 = desc0 * delta0_spatial_attention
            # delta1
            delta1_spatial_attention = delta1.mean(1) # mean
            delta1_spatial_attention = F.softmax(delta1_spatial_attention / 0.025, dim=-1)
            delta1_spatial_attention = delta1_spatial_attention.unsqueeze(1)# (B, 1, N)
            delta1 = desc1 * delta1_spatial_attention

            # residual connect
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1

