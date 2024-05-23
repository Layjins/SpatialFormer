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
    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class SpatialAttention(nn.Module):
    """ Scaled Dot-Product Spatial Attention """
    def __init__(self, mode='mean', using_softmax=False):
        super().__init__()
        self.mode = mode
        self.using_softmax = using_softmax

    def forward(self, q, k):
        # Element cosine similarity
        # q norm
        q_norm = torch.norm(q, p=2, dim=-1).unsqueeze(2).expand_as(q)
        q_norm = q.div(q_norm + 0.00001) # (B, h*w, c)
        # k norm
        k_norm = torch.norm(k, p=2, dim=-1).unsqueeze(2).expand_as(k)
        k_norm = k.div(k_norm + 0.00001) # (B, h*w, c)
        # similarity
        attn = torch.bmm(q_norm, k_norm.transpose(1, 2)) # (B, h*w, h*w)
        if self.mode == 'sum':
            cos_scores = torch.sum(attn, dim=-1) # (B, h*w)
        elif self.mode == 'mean':
            cos_scores = torch.mean(attn, dim=-1) # (B, h*w)
        elif self.mode == 'max':
            cos_scores = torch.max(attn, dim=-1)[0] # (B, h*w)
        # softmax
        if self.using_softmax:
            cos_scores = F.softmax(cos_scores / 0.025, dim=-1)
        # spatial attention
        cos_scores = cos_scores.unsqueeze(1) # (B, 1, h*w)
        output = q.transpose(1, 2) * cos_scores # (B, c, h*w)
        output = output.transpose(1, 2) # (B, h*w, c)
        return output, attn

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

class FFN_BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel=3, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(FFN_BasicBlock, self).__init__()
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
        self.relu = nn.ReLU(inplace=True)
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

##############################################

class ClassSpecific(nn.Module):
    def __init__(self, feature_dim, num_queries=100, num_heads=1, FFN_method='MLP'):
        super(ClassSpecific, self).__init__()
        self.feature_dim = feature_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.FFN_method = FFN_method
        # query embedding
        self.query_embed = nn.Embedding(self.num_queries, self.feature_dim)
        # attention
        self.with_W = False
        if self.num_heads > 1:
            self.attention = MultiHeadedAttention(self.num_heads, self.feature_dim, with_W=self.with_W) # (B, C, N)
            #self.mlp = MLP([self.feature_dim*2, self.feature_dim*2, self.feature_dim])
            #nn.init.constant_(self.mlp[-1].bias, 0.0)
            self.FFN = FFN_MLP(self.feature_dim, d_ffn=self.feature_dim*2)
        else:
            # with W
            if self.with_W:
                self.merge = nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=1)
                self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
            self.attention = ScaledDotProductAttention(temperature=np.power(self.feature_dim, 0.5)) # (B, N, C)
            # FFN
            if self.FFN_method == 'MLP':
                self.FFN = FFN_MLP(self.feature_dim, d_ffn=self.feature_dim*2)
            elif self.FFN_method == 'BasicBlock':
                self.FFN = FFN_BasicBlock(self.feature_dim, self.feature_dim)

    def forward(self, src):
        """
        Shape: 
        - src.size = B, c, h, w. B=batch size
        """
        B, c, h, w = src.size()
        assert c == self.feature_dim
        src = src.contiguous().view(B, c, h*w).transpose(1, 2) # (B, h*w, c)
        # feature interaction
        q = src # (B, h*w, c)
        query_embed_weight = self.query_embed.weight.unsqueeze(0).repeat(B,1,1) # (B, num_queries, c)
        k = query_embed_weight # (B, num_queries, c)
        v = query_embed_weight # (B, num_queries, c)
        if self.num_heads > 1:
            output, softmax_qk = self.attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)) # (B, c, h*w)
            ''' MLP
            output = self.mlp(torch.cat([src.transpose(1, 2), output], dim=1)) # (B, c, h*w)
            output = output.contiguous().view(B, c, h, w) # (B, c, h, w)
            '''
            #''' FFN_MLP
            output = output.transpose(1, 2) + src  # (B, h*w, c)
            output = self.FFN(output)  # (B, h*w, c)
            output = output.transpose(1, 2).contiguous().view(B, c, h, w) # (B, c, h, w)
            #'''   
        else:
            # with W
            if self.with_W:
                q, k, v = [l(x) for l, x in zip(self.proj, (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)))]
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

            output, _, softmax_qk = self.attention(q, k, v) # (B, h*w, c)
            # residual {add / product / cat+conv}
            output = output + src  # (B, h*w, c)
            # FFN
            if self.FFN_method == 'MLP':
                output = self.FFN(output)  # (B, h*w, c)
                output = output.transpose(1, 2).contiguous().view(B, c, h, w) # (B, c, h, w)
            elif self.FFN_method == 'BasicBlock':
                output = self.FFN(output.transpose(1, 2).contiguous().view(B, c, h, w)) # (B, c, h, w)
            else:
                output = output.transpose(1, 2).contiguous().view(B, c, h, w) # (B, c, h, w)
        return output, softmax_qk

class TaskSpecific(nn.Module):
    def __init__(self, feature_dim, num_queries, num_heads=1, FFN_method='MLP'):
        super(TaskSpecific, self).__init__()
        self.feature_dim = feature_dim
        self.num_queries = num_queries
        self.FFN_method = FFN_method
        # dataset embedding
        self.dataset_embed_s = nn.Embedding(self.num_queries['dataset'], self.feature_dim)
        self.attention = ScaledDotProductAttention(temperature=np.power(self.feature_dim, 0.5))
        # query embedding
        self.query_embed = nn.Embedding(self.num_queries['query'], self.feature_dim)
        self.attention_cross = ScaledDotProductAttention(temperature=np.power(self.feature_dim, 0.5))
        # query FFN
        if self.FFN_method == 'MLP':
            self.FFN = FFN_MLP(self.feature_dim)
        elif self.FFN_method == 'BasicBlock':
            self.FFN = FFN_BasicBlock(self.feature_dim, self.feature_dim)

    def forward(self, support_feat, query_feat):
        """
        Shape: 
        - support_feat.size = b, n1, c, h, w
        - query_feat.size = b, n2, c, h, w
        """
        b, n1, c, h, w = support_feat.size()
        n2 = query_feat.size(1)
        assert c == self.feature_dim
        assert n1 == self.num_queries['query']
        # support_proto
        support_proto = support_feat.view(*support_feat.size()[:3], -1).mean(3) # b, n1, c
        support_proto = support_proto.unsqueeze(1).repeat(1,n2,1,1) # b, n2, n1, c
        support_proto = support_proto.contiguous().view(b*n2, n1, c)
        #''' support_proto_att
        q_s = support_proto # (b*n2, n1, c)
        dataset_embed_s_weight = self.dataset_embed_s.weight.unsqueeze(0).repeat(b*n2,1,1) # (b*n2, num_queries['support'], c)
        k_s = v_s = dataset_embed_s_weight
        support_proto_att, _, _ = self.attention(q_s, k_s, v_s) # (b*n2, n1, c)
        #'''
        #support_proto_att = support_proto

        # query_feat
        query_feat = query_feat.contiguous().view(b*n2, c, h*w)
        query_feat = query_feat.transpose(1, 2) # (b*n2, h*w, c)
        ''' query_feat_att
        query_feat_proto = query_feat.mean(1).unsqueeze(1) # (b*n2, 1, c)
        q_q = query_feat_proto # (b*n2, 1, c)
        k_q = v_q = query_feat # (b*n2, h*w, c)
        query_feat_proto_att, _, _ = self.attention(q_q, k_q, v_q) # (b*n2, 1, c)
        query_feat_att = query_feat + query_feat_proto_att.repeat(1,h*w,1) # (b*n2, h*w, c)
        '''
        query_feat_att = query_feat

        # query_feat cross interaction
        q_cross_q = query_feat_att # (b*n2, h*w, c)
        k_cross_q = support_proto_att # (b*n2, n1, c)
        v_cross_q = self.query_embed.weight.unsqueeze(0).repeat(b*n2,1,1) # (b*n2, n1, c)
        #k_cross_q = v_cross_q = support_proto_att + self.query_embed.weight.unsqueeze(0).repeat(b*n2,1,1) # (b*n2, n1, c)
        #k_cross_q = v_cross_q = support_proto_att # (b*n2, n1, c)
        query_feat_cross_att, _, _ = self.attention_cross(q_cross_q, k_cross_q, v_cross_q) # (b*n2, h*w, c)
        # residual {add / product / cat+conv}
        query_feat_cross_att = query_feat_cross_att + query_feat_att  # (b*n2, h*w, c)
        #query_feat_cross_att = (query_feat_cross_att + 1) * query_feat_att  # (b*n2, h*w, c)
        # FFN
        if self.FFN_method == 'MLP':
            query_feat_cross_att = self.FFN(query_feat_cross_att) # (b*n2, h*w, c)
            query_feat_cross_att = query_feat_cross_att.transpose(1, 2).contiguous().view(b, n2, c, h, w) # (b, n2, c, h, w)
        elif self.FFN_method == 'BasicBlock':
            query_feat_cross_att = self.FFN(query_feat_cross_att.transpose(1, 2).contiguous().view(b*n2, c, h, w)) # (b*n2, c, h, w)
            query_feat_cross_att = query_feat_cross_att.contiguous().view(b, n2, c, h, w) # (b, n2, c, h, w)
        else:
            query_feat_cross_att = query_feat_cross_att.transpose(1, 2).contiguous().view(b, n2, c, h, w) # (b, n2, c, h, w)
        return query_feat_cross_att

class transformer_fuse_block(nn.Module):
    def __init__(self, feature_dim, num_heads=1):
        super(transformer_fuse_block, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        # attention
        self.with_W = False
        if self.num_heads > 1:
            self.attention = MultiHeadedAttention(self.num_heads, self.feature_dim, with_W=self.with_W) # (B, C, N)
            self.FFN = FFN_MLP(self.feature_dim, d_ffn=self.feature_dim*2)
        else:
            # with W
            if self.with_W:
                self.merge = nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=1)
                self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
            self.attention = ScaledDotProductAttention(temperature=np.power(self.feature_dim, 0.5)) # (B, N, C)
            # FFN
            self.FFN = FFN_MLP(self.feature_dim, d_ffn=self.feature_dim*2)

    def forward(self, src, base_embedded_prototype):
        """
        Shape: 
        - src.size = B, n, c. B=batch size
        """
        B, n, c = src.size()
        assert c == self.feature_dim
        # feature interaction
        q = src # (B, n, c)
        k = base_embedded_prototype # (B, base_class_num, c)
        v = base_embedded_prototype # (B, base_class_num, c)
        if self.num_heads > 1:
            output, softmax_qk = self.attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)) # (B, c, n)
            # FFN_MLP
            output = output.transpose(1, 2) + src  # (B, n, c)
            output = self.FFN(output)  # (B, n, c)
        else:
            # with W
            if self.with_W:
                q, k, v = [l(x) for l, x in zip(self.proj, (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)))]
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

            output, _, softmax_qk = self.attention(q, k, v) # (B, n, c)
            output = output + src  # (B, n, c)
            # FFN
            output = self.FFN(output)  # (B, n, c)
        return output

class tSF(nn.Module):
    def __init__(self, feature_dim, num_queries=100, num_novel_queries=0, num_heads=1, FFN_method='MLP', base_embedded_prototype_fuse=None):
        super(tSF, self).__init__()
        self.feature_dim = feature_dim
        self.num_queries = num_queries
        self.num_novel_queries = num_novel_queries
        self.num_heads = num_heads
        self.FFN_method = FFN_method
        self.base_embedded_prototype_fuse = base_embedded_prototype_fuse # 'add', 'transformer'
        # query embedding
        self.query_embed = nn.Embedding(self.num_queries, self.feature_dim)
        if self.num_novel_queries > 0:
            self.novel_query_embed = nn.Embedding(self.num_novel_queries, self.feature_dim)
        # base_embedded_prototype_fuse
        if self.base_embedded_prototype_fuse != None:
            self.fuse_conv_k = nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=1)
            self.fuse_conv_v = nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=1)
            if self.base_embedded_prototype_fuse == 'transformer':
                self.transformer_fuse_block = transformer_fuse_block(feature_dim=self.feature_dim, num_heads=self.num_heads)
        # attention
        self.with_W = False
        if self.num_heads > 1:
            self.attention = MultiHeadedAttention(self.num_heads, self.feature_dim, with_W=self.with_W) # (B, C, N)
            #self.mlp = MLP([self.feature_dim*2, self.feature_dim*2, self.feature_dim])
            #nn.init.constant_(self.mlp[-1].bias, 0.0)
            self.FFN = FFN_MLP(self.feature_dim, d_ffn=self.feature_dim*2)
        else:
            # with W
            if self.with_W:
                self.merge = nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=1)
                self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
            self.attention = ScaledDotProductAttention(temperature=np.power(self.feature_dim, 0.5)) # (B, N, C)
            # FFN
            if self.FFN_method == 'MLP':
                self.FFN = FFN_MLP(self.feature_dim, d_ffn=self.feature_dim*2)
            elif self.FFN_method == 'BasicBlock':
                self.FFN = FFN_BasicBlock(self.feature_dim, self.feature_dim)

    def forward(self, src, using_novel_query=False, base_embedded_prototype=None):
        """
        Shape: 
        - src.size = B, c, h, w. B=batch size
        - base_embedded_prototype.size = base_class_num, c.
        """
        B, c, h, w = src.size()
        assert c == self.feature_dim
        src = src.contiguous().view(B, c, h*w).transpose(1, 2) # (B, h*w, c)
        # feature interaction
        q = src # (B, h*w, c)
        query_embed_weight = self.query_embed.weight.unsqueeze(0).repeat(B,1,1) # (B, num_queries, c)
        if using_novel_query and self.num_novel_queries > 0:
            novel_query_embed_weight = self.novel_query_embed.weight.unsqueeze(0).repeat(B,1,1) # (B, num_novel_queries, c)
            query_embed_weight = torch.cat((query_embed_weight, novel_query_embed_weight), dim=1)
        k = query_embed_weight # (B, num_queries, c)
        v = query_embed_weight # (B, num_queries, c)
        if (self.base_embedded_prototype_fuse != None) and (base_embedded_prototype != None):
            if self.base_embedded_prototype_fuse == 'add':
                # add fusion
                base_class_num, base_class_c = base_embedded_prototype.size()
                # conv
                base_embedded_prototype = base_embedded_prototype.transpose(0, 1).view(1, base_class_c, base_class_num)
                base_embedded_prototype_k = self.fuse_conv_k(base_embedded_prototype)
                base_embedded_prototype_v = self.fuse_conv_v(base_embedded_prototype)
                base_embedded_prototype_k = base_embedded_prototype_k.transpose(1, 2).view(base_class_num, base_class_c)
                base_embedded_prototype_v = base_embedded_prototype_v.transpose(1, 2).view(base_class_num, base_class_c)
                # fusion
                _, num_queries, num_queries_c = query_embed_weight.size()
                assert num_queries_c == base_class_c
                assert num_queries == base_class_num
                k_base_embedded_prototype = base_embedded_prototype_k.repeat(B,1,1) # (B, base_class_num, c)
                v_base_embedded_prototype = base_embedded_prototype_v.repeat(B,1,1) # (B, base_class_num, c)
                k = k + k_base_embedded_prototype # (B, num_queries, c)
                v = v + v_base_embedded_prototype # (B, num_queries, c)
            elif self.base_embedded_prototype_fuse == 'transformer':
                # transformer fusion
                base_class_num, base_class_c = base_embedded_prototype.size()
                # conv
                base_embedded_prototype = base_embedded_prototype.transpose(0, 1).view(1, base_class_c, base_class_num)
                base_embedded_prototype_k = self.fuse_conv_k(base_embedded_prototype)
                base_embedded_prototype_v = self.fuse_conv_v(base_embedded_prototype)
                base_embedded_prototype_k = base_embedded_prototype_k.transpose(1, 2).view(base_class_num, base_class_c)
                base_embedded_prototype_v = base_embedded_prototype_v.transpose(1, 2).view(base_class_num, base_class_c)
                # fusion
                k_base_embedded_prototype = base_embedded_prototype_k.repeat(B,1,1) # (B, base_class_num, c)
                v_base_embedded_prototype = base_embedded_prototype_v.repeat(B,1,1) # (B, base_class_num, c)
                k = self.transformer_fuse_block(k, k_base_embedded_prototype) # (B, num_queries, c)
                v = self.transformer_fuse_block(v, v_base_embedded_prototype) # (B, num_queries, c)
        if self.num_heads > 1:
            output, softmax_qk = self.attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)) # (B, c, h*w)
            ''' MLP
            output = self.mlp(torch.cat([src.transpose(1, 2), output], dim=1)) # (B, c, h*w)
            output = output.contiguous().view(B, c, h, w) # (B, c, h, w)
            '''
            #''' FFN_MLP
            output = output.transpose(1, 2) + src  # (B, h*w, c)
            output = self.FFN(output)  # (B, h*w, c)
            output = output.transpose(1, 2).contiguous().view(B, c, h, w) # (B, c, h, w)
            #'''   
        else:
            # with W
            if self.with_W:
                q, k, v = [l(x) for l, x in zip(self.proj, (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)))]
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)

            output, _, softmax_qk = self.attention(q, k, v) # (B, h*w, c)
            # residual {add / product / cat+conv}
            output = output + src  # (B, h*w, c)
            # FFN
            if self.FFN_method == 'MLP':
                output = self.FFN(output)  # (B, h*w, c)
                output = output.transpose(1, 2).contiguous().view(B, c, h, w) # (B, c, h, w)
            elif self.FFN_method == 'BasicBlock':
                output = self.FFN(output.transpose(1, 2).contiguous().view(B, c, h, w)) # (B, c, h, w)
            else:
                output = output.transpose(1, 2).contiguous().view(B, c, h, w) # (B, c, h, w)
        # return output, softmax_qk
        return output
        

class tSF_stacker(nn.Module):
    def __init__(self, feature_dim, num_queries=100, num_heads=1, FFN_method='MLP', layer_num=1):
        super(tSF_stacker, self).__init__()
        tSF_stacker_layers = []
        for i in range(layer_num):
            tSF_stacker_block = tSF(feature_dim=feature_dim, num_queries=num_queries, num_heads=num_heads, FFN_method=FFN_method)
            tSF_stacker_layers.append(tSF_stacker_block)
        self.tSF_stacker_layers = nn.Sequential(*tSF_stacker_layers)

    def forward(self, src):
        """
        Shape: 
        - src.size = B, c, h, w. B=batch size
        """
        src = self.tSF_stacker_layers(src)  # (B, c, h, w)
        return src

class tSF_novel(nn.Module):
    def __init__(self, feature_dim, num_queries=100, num_novel_queries=0, num_heads=1, FFN_method='MLP', layer_num=1):
        super(tSF_novel, self).__init__()
        self.tSF_novel_block = tSF(feature_dim=feature_dim, num_queries=num_queries, num_novel_queries=num_novel_queries, num_heads=num_heads, FFN_method=FFN_method)

    def forward(self, src, using_novel_query=False):
        """
        Shape: 
        - src.size = B, c, h, w. B=batch size
        """
        src = self.tSF_novel_block(src, using_novel_query=using_novel_query)  # (B, c, h, w)
        return src

class tSF_encoder_block(nn.Module):
    def __init__(self, feature_dim, num_heads=1, FFN_method='MLP', conv_base_embedded_prototype=False, spatial_atte_transformer=False):
        super(tSF_encoder_block, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.FFN_method = FFN_method
        self.conv_base_embedded_prototype = conv_base_embedded_prototype
        self.spatial_atte_transformer = spatial_atte_transformer
        # base_embedded_prototype
        if self.conv_base_embedded_prototype == True:
            self.conv_k = nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=1)
            self.conv_v = nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=1)
        # attention
        if self.spatial_atte_transformer:
            # self.attention = SpatialAttention(mode='sum', using_softmax=False) # (B, N, C)
            self.attention = SpatialAttention(mode='mean', using_softmax=False) # (B, N, C)
            # self.attention = SpatialAttention(mode='max', using_softmax=False) # (B, N, C)
            # FFN
            self.FFN = FFN_MLP(self.feature_dim, d_ffn=self.feature_dim*2)
        else:
            self.with_W = False
            if self.num_heads > 1:
                self.attention = MultiHeadedAttention(self.num_heads, self.feature_dim, with_W=self.with_W) # (B, C, N)
                #self.mlp = MLP([self.feature_dim*2, self.feature_dim*2, self.feature_dim])
                #nn.init.constant_(self.mlp[-1].bias, 0.0)
                self.FFN = FFN_MLP(self.feature_dim, d_ffn=self.feature_dim*2)
            else:
                # with W
                if self.with_W:
                    self.merge = nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=1)
                    self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
                self.attention = ScaledDotProductAttention(temperature=np.power(self.feature_dim, 0.5)) # (B, N, C)
                # FFN
                if self.FFN_method == 'MLP':
                    self.FFN = FFN_MLP(self.feature_dim, d_ffn=self.feature_dim*2)
                elif self.FFN_method == 'BasicBlock':
                    self.FFN = FFN_BasicBlock(self.feature_dim, self.feature_dim)

    def forward(self, src, query_embed=None, query_feat=None, base_embedded_prototype=None, support_prototype=None):
        """
        Shape: 
        - src.size = B, c, h, w. B = batch size.
        - query_embed.size = num_queries, c.
        - query_feat.size = B, c, h, w.
        - base_embedded_prototype.size = base_class_num, c.
        - support_prototype.size = b, support_class_num, c.
        """
        B, c, h, w = src.size()
        assert c == self.feature_dim
        src = src.contiguous().view(B, c, h*w).transpose(1, 2) # (B, h*w, c)
        # feature interaction
        q = src # (B, h*w, c)
        # k, v
        if query_embed != None:
            query_embed_weight = query_embed.weight.unsqueeze(0).repeat(B,1,1) # (B, num_queries, c)
            k = query_embed_weight # (B, num_queries, c)
            v = query_embed_weight # (B, num_queries, c)
        elif base_embedded_prototype != None:
            base_class_num, base_class_c = base_embedded_prototype.size()
            # conv
            base_embedded_prototype = base_embedded_prototype.transpose(0, 1).view(1, base_class_c, base_class_num)
            if self.conv_base_embedded_prototype == True:
                base_embedded_prototype_k = self.conv_k(base_embedded_prototype)
                base_embedded_prototype_v = self.conv_v(base_embedded_prototype)
            else:
                base_embedded_prototype_k = base_embedded_prototype
                base_embedded_prototype_v = base_embedded_prototype
            base_embedded_prototype_k = base_embedded_prototype_k.transpose(1, 2).view(base_class_num, base_class_c)
            base_embedded_prototype_v = base_embedded_prototype_v.transpose(1, 2).view(base_class_num, base_class_c)
            k = base_embedded_prototype_k.repeat(B,1,1) # (B, base_class_num, c)
            v = base_embedded_prototype_v.repeat(B,1,1) # (B, base_class_num, c)
        elif support_prototype != None:
            b, support_class_num, c = support_prototype.size()
            support_prototype = support_prototype.unsqueeze(1).repeat(1,B//b,1,1) # (b, B//b, support_class_num, c)
            support_prototype = support_prototype.view(B, support_class_num, c)
            k = support_prototype # (B, support_class_num, c)
            v = support_prototype # (B, support_class_num, c)
        elif query_feat != None:
            B, c1, h1, w1 = query_feat.size()
            query_feat = query_feat.contiguous().view(B, c1, h1*w1).transpose(1, 2) # (B, h1*w1, c1)
            k = query_feat # (B, h1*w1, c1)
            v = query_feat # (B, h1*w1, c1)
        # attention
        if self.spatial_atte_transformer:
            output, atten = self.attention(q, k) # (B, h*w, c)
            output = output + src  # (B, h*w, c)
            # FFN
            output = self.FFN(output)  # (B, h*w, c)
            output = output.transpose(1, 2).contiguous().view(B, c, h, w) # (B, c, h, w)
        else:
            if self.num_heads > 1:
                output, softmax_qk = self.attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)) # (B, c, h*w)
                ''' MLP
                output = self.mlp(torch.cat([src.transpose(1, 2), output], dim=1)) # (B, c, h*w)
                output = output.contiguous().view(B, c, h, w) # (B, c, h, w)
                '''
                #''' FFN_MLP
                output = output.transpose(1, 2) + src  # (B, h*w, c)
                output = self.FFN(output)  # (B, h*w, c)
                output = output.transpose(1, 2).contiguous().view(B, c, h, w) # (B, c, h, w)
                #'''   
            else:
                # with W
                if self.with_W:
                    q, k, v = [l(x) for l, x in zip(self.proj, (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)))]
                    q = q.transpose(1, 2)
                    k = k.transpose(1, 2)
                    v = v.transpose(1, 2)

                output, _, softmax_qk = self.attention(q, k, v) # (B, h*w, c)
                # residual {add / product / cat+conv}
                output = output + src  # (B, h*w, c)
                # FFN
                if self.FFN_method == 'MLP':
                    output = self.FFN(output)  # (B, h*w, c)
                    output = output.transpose(1, 2).contiguous().view(B, c, h, w) # (B, c, h, w)
                elif self.FFN_method == 'BasicBlock':
                    output = self.FFN(output.transpose(1, 2).contiguous().view(B, c, h, w)) # (B, c, h, w)
                else:
                    output = output.transpose(1, 2).contiguous().view(B, c, h, w) # (B, c, h, w)
        # return output, softmax_qk
        return output

class tSF_encoder(nn.Module):
    def __init__(self, feature_dim, num_queries=100, num_heads=1, FFN_method='MLP', layer_num=1):
        super(tSF_encoder, self).__init__()
        # query embedding
        self.query_embed = nn.Embedding(num_queries, feature_dim)
        # tSF
        tSF_stacker_layers = []
        for i in range(layer_num):
            tSF_stacker_block = tSF_encoder_block(feature_dim=feature_dim, num_heads=num_heads, FFN_method=FFN_method)
            tSF_stacker_layers.append(tSF_stacker_block)
        self.tSF_stacker_layers = nn.ModuleList(tSF_stacker_layers)

    def forward(self, src):
        """
        Shape: 
        - src.size = B, c, h, w. B=batch size
        """
        for layer in self.tSF_stacker_layers:
            src = layer(src, query_embed=self.query_embed)  # (B, c, h, w)
        return src

class tSF_tPF(nn.Module):
    def __init__(self, feature_dim, num_queries=100, num_heads=1, FFN_method='MLP'):
        super(tSF_tPF, self).__init__()
        # tSF
        self.tSF_block = tSF_encoder_block(feature_dim=feature_dim, num_heads=num_heads, FFN_method=FFN_method)
        self.query_embed = nn.Embedding(num_queries, feature_dim) # query embedding
        # tPF
        self.tPF_block = tSF_encoder_block(feature_dim=feature_dim, num_heads=num_heads, FFN_method=FFN_method, conv_base_embedded_prototype=True)

    def forward(self, src, base_embedded_prototype):
        """
        Shape: 
        - src.size = B, c, h, w. B=batch size
        """
        src_tSF = self.tSF_block(src, query_embed=self.query_embed)  # (B, c, h, w)
        src_tPF = self.tPF_block(src, base_embedded_prototype=base_embedded_prototype)  # (B, c, h, w)
        output = src_tSF + src_tPF
        return output

class tSF_plus(nn.Module):
    def __init__(self, feature_dim, mode='tSF_E', add_tSF=False, num_queries=100, num_heads=1, FFN_method='MLP', base_num_classes=64):
        super(tSF_plus, self).__init__()
        self.mode = mode
        self.add_tSF = add_tSF
        if self.mode == 'tSF_F':
            # transformer
            self.transformer_block = tSF_encoder_block(feature_dim=feature_dim, num_heads=num_heads, FFN_method=FFN_method)
        elif self.mode == 'tSF_E':
            # tSF
            self.tSF_block = tSF_encoder_block(feature_dim=feature_dim, num_heads=num_heads, FFN_method=FFN_method)
            self.query_embed = nn.Embedding(num_queries, feature_dim) # query embedding
        elif self.mode == 'tSF_E_Metric':
            # tSF MLP
            self.tSF_block = tSF_encoder_block(feature_dim=feature_dim, num_heads=num_heads, FFN_method=FFN_method)
            self.query_embed = nn.Embedding(num_queries, feature_dim) # query embedding
        elif self.mode == 'tSF_SP':
            # support-prototype
            self.tSF_SP_block = tSF_encoder_block(feature_dim=feature_dim, num_heads=num_heads, FFN_method=FFN_method)
        elif self.mode == 'tSF_BEP':
            # tPF (base embedded-prototype)
            self.tPF_block = tSF_encoder_block(feature_dim=feature_dim, num_heads=num_heads, FFN_method=FFN_method, conv_base_embedded_prototype=True)
        elif self.mode == 'tSF_BEP_SP':
            # support-prototype
            self.tSF_SP_block = tSF_encoder_block(feature_dim=feature_dim, num_heads=num_heads, FFN_method=FFN_method)
            # tPF (base embedded-prototype)
            self.tPF_block = tSF_encoder_block(feature_dim=feature_dim, num_heads=num_heads, FFN_method=FFN_method, conv_base_embedded_prototype=True)
        elif (self.mode == 'tSF_BEP_local') or (self.mode == 'tSF_BEP_global'):
            self.tSF_BEP_clasifier = nn.Conv2d(feature_dim, base_num_classes, kernel_size=1, bias=False)
            self.tSF_BEP_block = tSF_encoder_block(feature_dim=feature_dim, num_heads=num_heads, FFN_method=FFN_method, conv_base_embedded_prototype=True)
        elif self.add_tSF == True:
            # add_tSF
            self.add_tSF_block = tSF_encoder_block(feature_dim=feature_dim, num_heads=num_heads, FFN_method=FFN_method)
            self.add_query_embed = nn.Embedding(num_queries, feature_dim) # query embedding
        elif self.mode == 'SAT_F':
            # transformer
            self.SAT_transformer_block = tSF_encoder_block(feature_dim=feature_dim, num_heads=num_heads, FFN_method=FFN_method, spatial_atte_transformer=True)
        elif self.mode == 'SAT_E':
            # tSF
            self.SAT_block = tSF_encoder_block(feature_dim=feature_dim, num_heads=num_heads, FFN_method=FFN_method, spatial_atte_transformer=True)
            self.query_embed = nn.Embedding(num_queries, feature_dim) # query embedding
        elif self.mode == 'SAT_SP':
            # support-prototype
            self.SAT_SP_block = tSF_encoder_block(feature_dim=feature_dim, num_heads=num_heads, FFN_method=FFN_method, spatial_atte_transformer=True)
        elif self.mode == 'SAT_BEP':
            # tPF (base embedded-prototype)
            self.SAT_tPF_block = tSF_encoder_block(feature_dim=feature_dim, num_heads=num_heads, FFN_method=FFN_method, conv_base_embedded_prototype=True, spatial_atte_transformer=True)

    def forward(self, src, base_embedded_prototype=None, support_prototype=None, data_configs=None, tSF_BEP_classifier_train=False):
        """
        Shape: 
        - src.size = B, c, h, w. B=batch size, B=b*n1*n2
        - base_embedded_prototype.size = base_class_num, c.
        - support_prototype.size = b, support_class_num, c.
        """
        # attention with different modes
        if self.mode == 'tSF_F':
            output = self.transformer_block(src, query_feat=src)  # (B, c, h, w)
        elif self.mode == 'tSF_E':
            output = self.tSF_block(src, query_embed=self.query_embed)  # (B, c, h, w)
        elif self.mode == 'tSF_E_Metric':
            output = self.tSF_block(src, query_embed=self.query_embed)  # (B, c, h, w)
        elif self.mode == 'tSF_SP':
            output = self.tSF_SP_block(src, support_prototype=support_prototype)  # (B, c, h, w) 
        elif self.mode == 'tSF_BEP':
            output = self.tPF_block(src, base_embedded_prototype=base_embedded_prototype)  # (B, c, h, w)
        elif self.mode == 'tSF_BEP_SP':
            # tSF_SP
            tSF_SP_output = self.tSF_SP_block(src, support_prototype=support_prototype)  # (B, c, h, w)
            # tSF_BEP
            tPF_output = self.tPF_block(src, base_embedded_prototype=base_embedded_prototype)  # (B, c, h, w)
            # add
            output = tPF_output + tSF_SP_output
        elif (self.mode == 'tSF_BEP_local') or (self.mode == 'tSF_BEP_global'):
            # before
            if tSF_BEP_classifier_train == True:
                # classifier
                tSF_BEP_clasifier_res = self.tSF_BEP_clasifier(src) # (num_test, base_num_classes, h, w)
                return tSF_BEP_clasifier_res
            # base embedded-prototype
            class_weight = self.tSF_BEP_clasifier.weight.data # (base_num_classes, c, 1, 1)
            class_weight = class_weight.squeeze(2).squeeze(2) # (base_num_classes, c)
            # attention
            output = self.tSF_BEP_block(src, base_embedded_prototype=class_weight) # (B, c, h, w)
            # # after
            # if tSF_BEP_classifier_train == True:
            #     # classifier
            #     tSF_BEP_clasifier_res = self.tSF_BEP_clasifier(output) # (num_test, base_num_classes, h, w)
            #     return tSF_BEP_clasifier_res
        elif self.mode == 'SAT_F':
            output = self.SAT_transformer_block(src, query_feat=src)  # (B, c, h, w)
        elif self.mode == 'SAT_E':
            output = self.SAT_block(src, query_embed=self.query_embed)  # (B, c, h, w)
        elif self.mode == 'SAT_SP':
            output = self.SAT_SP_block(src, support_prototype=support_prototype)  # (B, c, h, w)
        elif self.mode == 'SAT_BEP':
            output = self.SAT_tPF_block(src, base_embedded_prototype=base_embedded_prototype)  # (B, c, h, w)
        
        # add_tSF
        if self.add_tSF == True:
            add_tSF_res = self.add_tSF_block(src, query_embed=self.add_query_embed) # (B, c, h, w)
            output = output + add_tSF_res

        return output # (B, c, h, w)

class SIQ_encoder(nn.Module):
    def __init__(self, feature_dim, num_queries=100, num_heads=1, FFN_method='MLP', layer_num=1):
        super(SIQ_encoder, self).__init__()
        # query embedding
        self.query_embed = nn.Embedding(num_queries, feature_dim)
        # tSF
        tSF_stacker_layers = []
        for i in range(layer_num):
            tSF_stacker_block = tSF_encoder_block(feature_dim=feature_dim, num_heads=num_heads, FFN_method=FFN_method)
            tSF_stacker_layers.append(tSF_stacker_block)
        self.tSF_stacker_layers = nn.ModuleList(tSF_stacker_layers)

    def forward(self, src):
        """
        Shape: 
        - src.size = B, c, h, w. B=batch size
        """
        output = src
        for i, layer in enumerate(self.tSF_stacker_layers):
            if i == 0:
                output = layer(src, query_embed=self.query_embed)  # (B, c, h, w)
            else:
                output = layer(src, query_feat=output)  # (B, c, h, w)
        return output


class tSF_T(nn.Module):
    def __init__(self, feature_dim, num_queries=100, num_heads=1, FFN_method='MLP', layer_num=1):
        super(tSF_T, self).__init__()
        self.layer_num = layer_num
        # query embedding
        self.query_embed = nn.Embedding(num_queries, feature_dim)
        # tSF
        tSF_stacker_layers = []
        for i in range(self.layer_num):
            tSF_stacker_block = tSF_encoder_block(feature_dim=feature_dim, num_heads=num_heads, FFN_method=FFN_method)
            tSF_stacker_layers.append(tSF_stacker_block)
        self.tSF_stacker_layers = nn.ModuleList(tSF_stacker_layers)
        # transformer
        transformer_layers = []
        for i in range(self.layer_num):
            transformer_block = tSF_encoder_block(feature_dim=feature_dim, num_heads=num_heads, FFN_method=FFN_method)
            transformer_layers.append(transformer_block)
        self.transformer_layers = nn.ModuleList(transformer_layers)

    def forward(self, src):
        """
        Shape: 
        - src.size = B, c, h, w. B=batch size
        """
        for i in range(self.layer_num):
            # tSF
            src = self.tSF_stacker_layers[i](src, query_embed=self.query_embed)  # (B, c, h, w)
            # transformer
            src = self.transformer_layers[i](src, query_feat=src)  # (B, c, h, w)
        return src


class tSF_T_tSF(nn.Module):
    def __init__(self, feature_dim, num_queries=100, num_heads=1, FFN_method='MLP', layer_num=1):
        super(tSF_T_tSF, self).__init__()
        self.layer_num = layer_num
        # query embedding
        self.query_embed = nn.Embedding(num_queries, feature_dim)
        # tSF
        tSF_stacker_layers = []
        for i in range(self.layer_num):
            tSF_stacker_block = tSF_encoder_block(feature_dim=feature_dim, num_heads=num_heads, FFN_method=FFN_method)
            tSF_stacker_layers.append(tSF_stacker_block)
        self.tSF_stacker_layers = nn.ModuleList(tSF_stacker_layers)
        # transformer
        transformer_layers = []
        for i in range(self.layer_num):
            transformer_block = tSF_encoder_block(feature_dim=feature_dim, num_heads=num_heads, FFN_method=FFN_method)
            transformer_layers.append(transformer_block)
        self.transformer_layers = nn.ModuleList(transformer_layers)
        # tSF2
        tSF_stacker_layers2 = []
        for i in range(self.layer_num):
            tSF_stacker_block2 = tSF_encoder_block(feature_dim=feature_dim, num_heads=num_heads, FFN_method=FFN_method)
            tSF_stacker_layers2.append(tSF_stacker_block2)
        self.tSF_stacker_layers2 = nn.ModuleList(tSF_stacker_layers2)

    def forward(self, src):
        """
        Shape: 
        - src.size = B, c, h, w. B=batch size
        """
        for i in range(self.layer_num):
            # tSF
            src = self.tSF_stacker_layers[i](src, query_embed=self.query_embed)  # (B, c, h, w)
            # transformer
            src = self.transformer_layers[i](src, query_feat=src)  # (B, c, h, w)
            # tSF2
            src = self.tSF_stacker_layers2[i](src, query_embed=self.query_embed)  # (B, c, h, w)
        return src

class tSF_BDC(nn.Module):
    def __init__(self, feature_dims, num_queries=100, num_heads=1, FFN_method='MLP', layer_num=1):
        super(tSF_BDC, self).__init__()
        self.layer_num = layer_num
        # query embedding
        self.query_embed = nn.Embedding(num_queries, feature_dims[0])
        # tSF
        tSF_stacker_layers = []
        for i in range(self.layer_num):
            tSF_stacker_block = tSF_encoder_block(feature_dim=feature_dims[0], num_heads=num_heads, FFN_method=FFN_method)
            tSF_stacker_layers.append(tSF_stacker_block)
        self.tSF_stacker_layers = nn.ModuleList(tSF_stacker_layers)
        # BDC
        BDC_layers = []
        for i in range(self.layer_num):
            BDC_block = BDC(is_vec=True, input_dim=feature_dims, dimension_reduction=None)
            BDC_layers.append(BDC_block)
        self.BDC_layers = nn.ModuleList(BDC_layers)

    def forward(self, src):
        """
        Shape: 
        - src.size = B, c, h, w. B=batch size
        """
        for i in range(self.layer_num):
            # tSF
            src = self.tSF_stacker_layers[i](src, query_embed=self.query_embed)  # (B, c, h, w)
            # BDC
            src = self.BDC_layers[i](src)  # (B, c, h, w)
        return src


class MSF(nn.Module):
    def __init__(self, feature_dim, num_queries=[1,5,10,20], num_heads=1, FFN_method='MLP'):
        super(MSF, self).__init__()
        self.num_queries = num_queries
        self.layer_num = len(self.num_queries)
        tSF_stacker_layers = []
        for i in range(self.layer_num):
            tSF_stacker_block = tSF(feature_dim=feature_dim, num_queries=self.num_queries[i], num_heads=num_heads, FFN_method=FFN_method)
            tSF_stacker_layers.append(tSF_stacker_block)
        self.tSF_stacker_layers = nn.ModuleList(tSF_stacker_layers)

    def forward(self, src):
        """
        Shape: 
        - src.size = B, c, h, w. B=batch size
        """
        output = self.tSF_stacker_layers[0](src)  # (B, c, h, w)
        for i in range(self.layer_num):
            if i > 0:
                output = output + self.tSF_stacker_layers[i](src)  # (B, c, h, w)
        return output
