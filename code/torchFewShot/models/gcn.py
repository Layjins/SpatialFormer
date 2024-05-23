import torch
import torch.nn as nn
from torch.nn import functional as F


class DGCN(nn.Module):
    def __init__(self, in_features, out_features, num_nodes):
        super(DGCN, self).__init__()

        self.static_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))
        self.static_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features*2, num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)

    def forward_static_gcn(self, x):
        x = self.static_adj(x.transpose(1, 2))
        x = self.static_weight(x.transpose(1, 2))
        return x

    def forward_construct_dynamic_graph(self, x):
        ### Model global representations ###
        x_glb = self.gap(x)
        x_glb = self.conv_global(x_glb)
        x_glb = self.bn_global(x_glb)
        x_glb = self.relu(x_glb)
        x_glb = x_glb.expand(x_glb.size(0), x_glb.size(1), x.size(2))
        
        ### Construct the dynamic correlation matrix ###
        x = torch.cat((x_glb, x), dim=1)
        dynamic_adj = self.conv_create_co_mat(x)
        dynamic_adj = torch.sigmoid(dynamic_adj)
        return dynamic_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):
        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        """ D-GCN module

        Shape: 
        - Input: (B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        out_static = self.forward_static_gcn(x)
        x = x + out_static # residual
        dynamic_adj = self.forward_construct_dynamic_graph(x)
        x = self.forward_dynamic_gcn(x, dynamic_adj)
        return x

class DGCN_self(nn.Module):
    def __init__(self, in_features, out_features, num_nodes):
        super(DGCN_self, self).__init__()

        self.static_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))
        self.static_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))
        #self.static_weight = nn.Sequential(
        #    nn.Conv1d(in_features, 1, 1),
        #    nn.LeakyReLU(0.2))

        self.conv_to_1 = nn.Conv1d(in_features, 1, 1)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features*2, num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)

    def forward_static_gcn(self, x):
        x = self.static_adj(x.transpose(1, 2))
        x = self.static_weight(x.transpose(1, 2))
        return x

    def forward_construct_dynamic_graph(self, x):
        ### Model global representations ###
        x_glb = self.gap(x)
        x_glb = self.conv_global(x_glb)
        x_glb = self.bn_global(x_glb)
        x_glb = self.relu(x_glb)
        x_glb = x_glb.expand(x_glb.size(0), x_glb.size(1), x.size(2))
        
        ### Construct the dynamic correlation matrix ###
        x = torch.cat((x_glb, x), dim=1)
        dynamic_adj = self.conv_create_co_mat(x)
        dynamic_adj = torch.sigmoid(dynamic_adj)
        return dynamic_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):
        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        """ D-GCN module

        Shape: 
        - Input: (B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        out_static = self.forward_static_gcn(x)
        #x = x + out_static # residual
        #dynamic_adj = self.forward_construct_dynamic_graph(x)
        #x = self.forward_dynamic_gcn(x, dynamic_adj)
        # attention
        #out_static = x + out_static # residual
        spatial_attention = out_static.mean(1) # mean
        #spatial_attention = self.conv_to_1(out_static)
        spatial_attention = F.softmax(spatial_attention / 0.025, dim=-1) + 1
        spatial_attention = spatial_attention.unsqueeze(1)# (B, 1, N)
        x = x * spatial_attention
        return x

class SGCN(nn.Module):
    def __init__(self, in_features, out_features, num_nodes):
        super(SGCN, self).__init__()

        self.static_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))
        self.static_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.conv_global = nn.Conv1d(in_features, in_features, 1)
        self.bn_global = nn.BatchNorm1d(in_features)
        self.relu = nn.LeakyReLU(0.2)

        self.conv_create_co_mat = nn.Conv1d(in_features*2, num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)

    def forward_static_gcn(self, x):
        x = self.static_adj(x.transpose(1, 2))
        x = self.static_weight(x.transpose(1, 2))
        return x

    def forward_construct_dynamic_graph(self, x):
        ### Model global representations ###
        x_glb = self.gap(x)
        x_glb = self.conv_global(x_glb)
        x_glb = self.bn_global(x_glb)
        x_glb = self.relu(x_glb)
        x_glb = x_glb.expand(x_glb.size(0), x_glb.size(1), x.size(2))

        ### Construct the dynamic correlation matrix ###
        x = torch.cat((x_glb, x), dim=1)
        dynamic_adj = self.conv_create_co_mat(x)
        dynamic_adj = torch.sigmoid(dynamic_adj)
        return dynamic_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):
        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        """ D-GCN module

        Shape: 
        - Input: (B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        out_static = self.forward_static_gcn(x)
        x = x + out_static # residual
        dynamic_adj = self.forward_construct_dynamic_graph(x)
        out_dynamic = self.forward_dynamic_gcn(x, dynamic_adj)
        #x = x * out_dynamic
        # attention
        B, C, N = x.size()
        spatial_attention = torch.mean(out_dynamic, 1)
        spatial_attention = F.softmax(spatial_attention / 0.025, dim=-1) + 1
        spatial_attention = spatial_attention.unsqueeze(1)# (B, 1, N)
        x = x * spatial_attention
        return x

class SCGCN(nn.Module):
    def __init__(self, in_features, out_features, num_nodes):
        super(SCGCN, self).__init__()

        self.static_adj = nn.Sequential(
            nn.Conv1d(num_nodes, num_nodes, 1, bias=False),
            nn.LeakyReLU(0.2))
        self.static_weight = nn.Sequential(
            nn.Conv1d(in_features, out_features, 1),
            nn.LeakyReLU(0.2))

        self.relu = nn.LeakyReLU(0.2)
        self.conv_create_co_mat2 = nn.Conv1d(num_nodes, num_nodes, 1)
        self.conv_create_co_mat = nn.Conv1d(in_features, num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)

        # attention
        self.attention_conv_1 = nn.Conv1d(out_features, 1, 1)
        self.attention_conv_2 = nn.Sequential(
            nn.Conv1d(out_features, out_features//2, 1),
            nn.ReLU(),
            nn.Conv1d(out_features//2, out_features, 1),
            nn.ReLU()
        )
        self.attention_conv_3 = nn.Sequential(
            nn.Conv1d(out_features, out_features//2, 1),
            nn.ReLU(),
            nn.Conv1d(out_features//2, out_features, 1),
            nn.ReLU()
        )

    def forward_static_gcn(self, x):
        x = self.static_adj(x.transpose(1, 2))
        x = self.static_weight(x.transpose(1, 2))
        return x

    def forward_construct_dynamic_graph(self, x):
        ### Construct the dynamic correlation matrix ###
        dynamic_adj = self.conv_create_co_mat(x)
        dynamic_adj = torch.sigmoid(dynamic_adj)
        return dynamic_adj

    def forward_construct_correlation_graph(self, x):
        ### Construct the dynamic correlation matrix ###
        B, C, N = x.size()
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-12) # (B, C, N)
        x_norm_t = x_norm.transpose(1, 2) # (B, N, C)
        correlation_adj = torch.matmul(x_norm_t, x_norm)  # (B, N, N)
        correlation_adj = self.conv_create_co_mat2(correlation_adj)
        correlation_adj = torch.sigmoid(correlation_adj)
        # only cross attention
        fix_adj_self = torch.eye(N//2)
        fix_adj_cross = torch.ones(N//2, N//2)
        fix_adj_1 = torch.cat((fix_adj_self, fix_adj_cross), 1)
        fix_adj_2 = torch.cat((fix_adj_cross, fix_adj_self), 1)
        fix_adj = torch.cat((fix_adj_1, fix_adj_2), 0)
        correlation_adj = correlation_adj * fix_adj.cuda()
        return correlation_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):
        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        #x = torch.sigmoid(x)
        return x

    def forward(self, x):
        """ D-GCN module

        Shape: 
        - Input: (B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        #out_static = self.forward_static_gcn(x)
        #x = x + out_static # residual
        #dynamic_adj = self.forward_construct_dynamic_graph(x)
        dynamic_adj = self.forward_construct_correlation_graph(x)
        #print(dynamic_adj.shape) # torch.Size([B, N, N])
        out_dynamic = self.forward_dynamic_gcn(x, dynamic_adj) # (B, C, N)
        #x = out_dynamic
        #x = x + out_dynamic # residual

        # attention
        # spatial attention
        B, C, N = x.size()
        out_dynamic_2 = self.attention_conv_2(out_dynamic)
        spatial_attention = self.attention_conv_1(out_dynamic_2).squeeze(1)
        #spatial_attention = torch.mean(out_dynamic, 1)
        spatial_attention_1 = spatial_attention[:,:N//2]
        spatial_attention_2 = spatial_attention[:,N//2:]
        # softmax_hw
        spatial_attention_1 = F.softmax(spatial_attention_1 / 0.025, dim=-1) + 1
        spatial_attention_2 = F.softmax(spatial_attention_2 / 0.025, dim=-1) + 1
        spatial_attention = torch.cat((spatial_attention_1, spatial_attention_2), 1)
        # softmax_2
        #spatial_attention = torch.stack((spatial_attention_1, spatial_attention_2), 1) # (B, 2, N//2)
        #spatial_attention = F.softmax(spatial_attention / 0.025, dim=1) + 1
        #spatial_attention = spatial_attention.contiguous().view(B, N)
        # attention op
        spatial_attention = spatial_attention.unsqueeze(1)# (B, 1, N)
        x = x * spatial_attention # spatial attention

        '''
        # channel attention
        out_dynamic = self.attention_conv_3(out_dynamic)
        channel_attention_1 = out_dynamic[:,:,:N//2]
        channel_attention_2 = out_dynamic[:,:,N//2:]
        channel_attention_1 = torch.mean(channel_attention_1, 2)
        channel_attention_2 = torch.mean(channel_attention_2, 2)
        # softmax_c
        #channel_attention_1 = F.softmax(channel_attention_1 / 0.025, dim=-1)
        #channel_attention_2 = F.softmax(channel_attention_2 / 0.025, dim=-1)
        # softmax_2
        channel_attention = torch.stack((channel_attention_1, channel_attention_2), 1) # (B, 2, C)
        channel_attention = F.softmax(channel_attention / 0.025, dim=1)
        channel_attention_1 = channel_attention[:,0,:]
        channel_attention_2 = channel_attention[:,1,:]
        # attention op
        channel_attention_1 = channel_attention_1.unsqueeze(2).expand(B,C,N//2)
        channel_attention_2 = channel_attention_2.unsqueeze(2).expand(B,C,N//2)
        channel_attention = torch.cat((channel_attention_1, channel_attention_2), 2)
        x = x * (channel_attention + 1) # channel attention
        '''
        return x

class SCGCN_Loop(nn.Module):
    def __init__(self, in_features, out_features, num_nodes, loop=1):
        super(SCGCN_Loop, self).__init__()
        self.loop = loop
        self.relu = nn.LeakyReLU(0.2)
        self.conv_create_co_mat = nn.Conv1d(num_nodes, num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(in_features, out_features, 1)

        # attention
        self.attention_conv_1 = nn.Conv1d(out_features, 1, 1)
        self.attention_conv_2 = nn.Sequential(
            nn.Conv1d(out_features, out_features//2, 1),
            nn.ReLU(),
            nn.Conv1d(out_features//2, out_features, 1),
            nn.ReLU()
        )

    def forward_construct_correlation_graph(self, x, pre_probability):
        ### Construct the dynamic correlation matrix ###
        B, C, N = x.size()
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-12) # (B, C, N)
        x_norm_t = x_norm.transpose(1, 2) # (B, N, C)
        correlation_adj = torch.matmul(x_norm_t, x_norm) # (B, N, N)

        pre_probability = pre_probability.unsqueeze(1)
        pre_probability = pre_probability.unsqueeze(2)
        pre_probability = pre_probability.expand(B,N,N)
        correlation_adj = correlation_adj * pre_probability

        correlation_adj = self.conv_create_co_mat(correlation_adj)
        correlation_adj = torch.sigmoid(correlation_adj)
        # only cross attention
        fix_adj_self = torch.eye(N//2)
        fix_adj_cross = torch.ones(N//2, N//2)
        fix_adj_1 = torch.cat((fix_adj_self, fix_adj_cross), 1)
        fix_adj_2 = torch.cat((fix_adj_cross, fix_adj_self), 1)
        fix_adj = torch.cat((fix_adj_1, fix_adj_2), 0)
        correlation_adj = correlation_adj * fix_adj.cuda()
        return correlation_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):
        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        #x = torch.sigmoid(x)
        return x

    def forward(self, x, shape_param, pre_probability=None):
        """
        Shape: 
        - Input: x(B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        b, n2, n1, c, h, w = shape_param
        B, C, N = x.size() # B=b*n2*n1, C=c, N=h*w*2
        for loop_index in range(self.loop):
            # pre_probability
            if pre_probability == None:
                # init pre_probability
                pre_probability = torch.ones(B)
                pre_probability = pre_probability.cuda()

            # gcn
            dynamic_adj = self.forward_construct_correlation_graph(x, pre_probability) # (B, N, N)
            out_dynamic = self.forward_dynamic_gcn(x, dynamic_adj) # (B, C, N)

            # gcn attention
            # spatial attention
            out_dynamic_2 = self.attention_conv_2(out_dynamic)
            spatial_attention = self.attention_conv_1(out_dynamic_2).squeeze(1)
            spatial_attention_1 = spatial_attention[:,:N//2]
            spatial_attention_2 = spatial_attention[:,N//2:]
            # softmax_hw
            spatial_attention_1 = F.softmax(spatial_attention_1 / 0.025, dim=-1) + 1
            spatial_attention_2 = F.softmax(spatial_attention_2 / 0.025, dim=-1) + 1
            spatial_attention = torch.cat((spatial_attention_1, spatial_attention_2), 1)
            # attention op
            spatial_attention = spatial_attention.unsqueeze(1)# (B, 1, N)
            x = x * spatial_attention # spatial attention

            # get pre_probability
            # split features
            f_cat = x
            f_cat = f_cat.contiguous().view(b, n2, n1, c, h*w*2)
            ftrain = f_cat[:,:,:,:,:h*w]
            ftest = f_cat[:,:,:,:,h*w:]
            ftrain = ftrain.contiguous().view(b, n2, n1, c, h, w)
            ftest = ftest.contiguous().view(b, n2, n1, c, h, w)
            # pre
            ftrain = ftrain.mean(4)
            ftrain = ftrain.mean(4) # (b, n2, n1, c)
            ftest = ftest.mean(4)
            ftest = ftest.mean(4) # (b, n2, n1, c)
            ftest = F.normalize(ftest, p=2, dim=ftest.dim()-1, eps=1e-12)
            ftrain = F.normalize(ftrain, p=2, dim=ftrain.dim()-1, eps=1e-12)
            self.scale_cls = 1 # default=7
            scores_sum = torch.sum(ftest * ftrain, dim=-1)  # (b, n2, n1)
            scores = self.scale_cls * (scores_sum * scores_sum)
            #scores = self.scale_cls * scores_sum
            pre_probability = F.softmax(scores, dim=-1)
            #print(pre_probability)
            pre_probability = pre_probability.contiguous().view(b*n2*n1)

        return x, pre_probability

class DGC_CAM(nn.Module):
    def __init__(self, in_features, num_nodes, matirx_name='Mcross', scale_cls=7):
        super(DGC_CAM, self).__init__()
        self.matirx_name = matirx_name
        self.scale_cls = scale_cls
        self.relu = nn.LeakyReLU(0.2)
        self.conv_create_co_mat = nn.Conv1d(num_nodes, num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(in_features, 1, 1)

    def forward_construct_correlation_graph(self, x):
        ### Construct the dynamic correlation matrix ###
        B, C, N = x.size()
        # cosin similarity
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-12) # (B, C, N)
        x_norm_t = x_norm.transpose(1, 2) # (B, N, C)
        # Mcorr
        correlation_adj = torch.matmul(x_norm_t, x_norm) # (B, N, N)
        if self.matirx_name == 'Mcorr_0':
            # Mcorr_0
            fix_adj_self = torch.zeros(N//2, N//2)
            fix_adj_cross = torch.ones(N//2, N//2)
            fix_adj_1 = torch.cat((fix_adj_self, fix_adj_cross), 1)
            fix_adj_2 = torch.cat((fix_adj_cross, fix_adj_cross), 1)
            fix_adj = torch.cat((fix_adj_1, fix_adj_2), 0)
            correlation_adj = correlation_adj * fix_adj.cuda()
        elif self.matirx_name == 'Mcorr_1':
            # Mcorr_1
            fix_adj_self = torch.eye(N//2)
            fix_adj_cross = torch.ones(N//2, N//2)
            fix_adj_1 = torch.cat((fix_adj_self, fix_adj_cross), 1)
            fix_adj_2 = torch.cat((fix_adj_cross, fix_adj_cross), 1)
            fix_adj = torch.cat((fix_adj_1, fix_adj_2), 0)
            correlation_adj = correlation_adj * fix_adj.cuda()
        elif self.matirx_name == 'Mcross':
            # Mcross: only cross attention
            fix_adj_self = torch.eye(N//2)
            fix_adj_cross = torch.ones(N//2, N//2)
            fix_adj_1 = torch.cat((fix_adj_self, fix_adj_cross), 1)
            fix_adj_2 = torch.cat((fix_adj_cross, fix_adj_self), 1)
            fix_adj = torch.cat((fix_adj_1, fix_adj_2), 0)
            correlation_adj = correlation_adj * fix_adj.cuda()
        elif self.matirx_name == 'Mcross_0':
            # Mcross_0: only cross attention
            fix_adj_self = torch.zeros(N//2, N//2)
            fix_adj_cross = torch.ones(N//2, N//2)
            fix_adj_1 = torch.cat((fix_adj_self, fix_adj_cross), 1)
            fix_adj_2 = torch.cat((fix_adj_cross, fix_adj_self), 1)
            fix_adj = torch.cat((fix_adj_1, fix_adj_2), 0)
            correlation_adj = correlation_adj * fix_adj.cuda()
        elif self.matirx_name == 'Mself':
            # Mself: only self attention
            fix_adj_self = torch.ones(N//2, N//2)
            fix_adj_cross = torch.zeros(N//2, N//2)
            fix_adj_1 = torch.cat((fix_adj_self, fix_adj_cross), 1)
            fix_adj_2 = torch.cat((fix_adj_cross, fix_adj_self), 1)
            fix_adj = torch.cat((fix_adj_1, fix_adj_2), 0)
            correlation_adj = correlation_adj * fix_adj.cuda()
        elif self.matirx_name == 'Mself_0':
            # Mself_0: only self attention
            fix_adj_self = torch.ones(N//2, N//2)
            fix_adj_cross = torch.zeros(N//2, N//2)
            fix_adj_1 = torch.cat((fix_adj_cross, fix_adj_cross), 1)
            fix_adj_2 = torch.cat((fix_adj_cross, fix_adj_self), 1)
            fix_adj = torch.cat((fix_adj_1, fix_adj_2), 0)
            correlation_adj = correlation_adj * fix_adj.cuda()
        elif self.matirx_name == 'Mself_1':
            # Mself_1: only self attention
            fix_adj_eye = torch.eye(N//2)
            fix_adj_self = torch.ones(N//2, N//2)
            fix_adj_cross = torch.zeros(N//2, N//2)
            fix_adj_1 = torch.cat((fix_adj_eye, fix_adj_cross), 1)
            fix_adj_2 = torch.cat((fix_adj_cross, fix_adj_self), 1)
            fix_adj = torch.cat((fix_adj_1, fix_adj_2), 0)
            correlation_adj = correlation_adj * fix_adj.cuda()
        elif self.matirx_name == 'Munit':
            # Munit: unit adj
            fix_adj_self = torch.eye(N//2)
            fix_adj_cross = torch.zeros(N//2, N//2)
            fix_adj_1 = torch.cat((fix_adj_self, fix_adj_cross), 1)
            fix_adj_2 = torch.cat((fix_adj_cross, fix_adj_self), 1)
            fix_adj = torch.cat((fix_adj_1, fix_adj_2), 0)
            correlation_adj = correlation_adj * fix_adj.cuda()
        
        return correlation_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):
        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        """
        Shape: 
        - Input: x(B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        B, C, N = x.size() # B=b*n2*n1, C=c, N=h*w*2
        ## gcn ##
        dynamic_adj = self.forward_construct_correlation_graph(x) # (B, N, N)
        #print(dynamic_adj[0])
        out_dynamic = self.forward_dynamic_gcn(x, dynamic_adj) # (B, 1, N)

        ## gcn attention ##
        spatial_attention = out_dynamic.squeeze(1)
        spatial_attention_1 = spatial_attention[:,:N//2]
        spatial_attention_2 = spatial_attention[:,N//2:]
        # softmax_hw
        spatial_attention_1 = F.softmax(spatial_attention_1 / 0.025, dim=-1) + 1
        spatial_attention_2 = F.softmax(spatial_attention_2 / 0.025, dim=-1) + 1
        # sigmoid
        #spatial_attention_1 = torch.sigmoid(spatial_attention_1) + 1
        #spatial_attention_2 = torch.sigmoid(spatial_attention_2) + 1
        # attention op
        spatial_attention = torch.cat((spatial_attention_1, spatial_attention_2), 1)
        spatial_attention = spatial_attention.unsqueeze(1)# (B, 1, N)
        x = x * spatial_attention # spatial attention
        return x

class SCGCN_self(nn.Module):
    def __init__(self, in_features, num_nodes):
        super(SCGCN_self, self).__init__()
        self.relu = nn.LeakyReLU(0.2)
        self.dynamic_weight = nn.Conv1d(in_features, 1, 1)

    def forward_construct_correlation_graph(self, x):
        ### Construct the dynamic correlation matrix ###
        B, C, N = x.size()
        # cosin similarity
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-12) # (B, C, N)
        x_norm_t = x_norm.transpose(1, 2) # (B, N, C)
        # Mcorr
        correlation_adj = torch.matmul(x_norm_t, x_norm) # (B, N, N)
        return correlation_adj

    def forward_dynamic_gcn(self, x, dynamic_adj):
        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        """
        Shape: 
        - Input: x(B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        B, C, N = x.size()
        ## gcn ##
        dynamic_adj = self.forward_construct_correlation_graph(x) # (B, N, N)
        out_dynamic = self.forward_dynamic_gcn(x, dynamic_adj) # (B, 1, N)

        ## gcn attention ##
        spatial_attention = out_dynamic.squeeze(1)
        # softmax_hw
        spatial_attention = F.softmax(spatial_attention / 0.025, dim=-1) + 1
        # attention op
        spatial_attention = spatial_attention.unsqueeze(1)# (B, 1, N)
        x = x * spatial_attention # spatial attention
        return x

class DGC_CAM2(nn.Module):
    def __init__(self, in_features, num_nodes, matirx_name='Mcross', scale_cls=7):
        super(DGC_CAM2, self).__init__()
        self.matirx_name = matirx_name
        self.scale_cls = scale_cls
        self.relu = nn.LeakyReLU(0.2)
        self.conv_create_co_mat = nn.Conv1d(num_nodes, num_nodes, 1)
        self.dynamic_weight = nn.Conv1d(in_features, in_features, 1)
        # attention
        self.attention_conv_1 = nn.Conv1d(in_features, 1, 1)

    def forward_construct_correlation_graph(self, x):
        ### Construct the dynamic correlation matrix ###
        B, C, N = x.size()
        # cosin similarity
        x_norm = F.normalize(x, p=2, dim=1, eps=1e-12) # (B, C, N)
        x_norm_t = x_norm.transpose(1, 2) # (B, N, C)
        # Mcorr
        correlation_adj = torch.matmul(x_norm_t, x_norm) # (B, N, N)
        # conv for correlation_adj
        #correlation_adj = self.conv_create_co_mat(correlation_adj)
        #correlation_adj = torch.sigmoid(correlation_adj)
        # matrixs
        if self.matirx_name == 'Mcorr_0':
            # Mcorr_0
            fix_adj_self = torch.zeros(N//2, N//2)
            fix_adj_cross = torch.ones(N//2, N//2)
            fix_adj_1 = torch.cat((fix_adj_self, fix_adj_cross), 1)
            fix_adj_2 = torch.cat((fix_adj_cross, fix_adj_cross), 1)
            fix_adj = torch.cat((fix_adj_1, fix_adj_2), 0)
            correlation_adj = correlation_adj * fix_adj.cuda()
        elif self.matirx_name == 'Mcorr_1':
            # Mcorr_1
            fix_adj_self = torch.eye(N//2)
            fix_adj_cross = torch.ones(N//2, N//2)
            fix_adj_1 = torch.cat((fix_adj_self, fix_adj_cross), 1)
            fix_adj_2 = torch.cat((fix_adj_cross, fix_adj_cross), 1)
            fix_adj = torch.cat((fix_adj_1, fix_adj_2), 0)
            correlation_adj = correlation_adj * fix_adj.cuda()
        elif self.matirx_name == 'Mcross':
            # Mcross: only cross attention
            fix_adj_self = torch.eye(N//2)
            fix_adj_cross = torch.ones(N//2, N//2)
            fix_adj_1 = torch.cat((fix_adj_self, fix_adj_cross), 1)
            fix_adj_2 = torch.cat((fix_adj_cross, fix_adj_self), 1)
            fix_adj = torch.cat((fix_adj_1, fix_adj_2), 0)
            correlation_adj = correlation_adj * fix_adj.cuda()
        elif self.matirx_name == 'Mcross_1':
            # Mcross_1: only cross attention
            correlation_adj[:,:N//2,:N//2] = torch.ones(N//2, N//2)
            correlation_adj[:,N//2:,N//2:] = torch.ones(N//2, N//2)
        elif self.matirx_name == 'Mcross_0':
            # Mcross_0: only cross attention
            fix_adj_self = torch.zeros(N//2, N//2)
            fix_adj_cross = torch.ones(N//2, N//2)
            fix_adj_1 = torch.cat((fix_adj_self, fix_adj_cross), 1)
            fix_adj_2 = torch.cat((fix_adj_cross, fix_adj_self), 1)
            fix_adj = torch.cat((fix_adj_1, fix_adj_2), 0)
            correlation_adj = correlation_adj * fix_adj.cuda()
        elif self.matirx_name == 'Mself':
            # Mself: only self attention
            fix_adj_self = torch.ones(N//2, N//2)
            fix_adj_cross = torch.zeros(N//2, N//2)
            fix_adj_1 = torch.cat((fix_adj_self, fix_adj_cross), 1)
            fix_adj_2 = torch.cat((fix_adj_cross, fix_adj_self), 1)
            fix_adj = torch.cat((fix_adj_1, fix_adj_2), 0)
            correlation_adj = correlation_adj * fix_adj.cuda()
        elif self.matirx_name == 'Mself_0':
            # Mself_0: only self attention
            fix_adj_self = torch.ones(N//2, N//2)
            fix_adj_cross = torch.zeros(N//2, N//2)
            fix_adj_1 = torch.cat((fix_adj_cross, fix_adj_cross), 1)
            fix_adj_2 = torch.cat((fix_adj_cross, fix_adj_self), 1)
            fix_adj = torch.cat((fix_adj_1, fix_adj_2), 0)
            correlation_adj = correlation_adj * fix_adj.cuda()
        elif self.matirx_name == 'Mself_1':
            # Mself_1: only self attention
            fix_adj_eye = torch.eye(N//2)
            fix_adj_self = torch.ones(N//2, N//2)
            fix_adj_cross = torch.zeros(N//2, N//2)
            fix_adj_1 = torch.cat((fix_adj_eye, fix_adj_cross), 1)
            fix_adj_2 = torch.cat((fix_adj_cross, fix_adj_self), 1)
            fix_adj = torch.cat((fix_adj_1, fix_adj_2), 0)
            correlation_adj = correlation_adj * fix_adj.cuda()
        elif self.matirx_name == 'Munit':
            # Munit: unit adj
            fix_adj_self = torch.eye(N//2)
            fix_adj_cross = torch.zeros(N//2, N//2)
            fix_adj_1 = torch.cat((fix_adj_self, fix_adj_cross), 1)
            fix_adj_2 = torch.cat((fix_adj_cross, fix_adj_self), 1)
            fix_adj = torch.cat((fix_adj_1, fix_adj_2), 0)
            correlation_adj = correlation_adj * fix_adj.cuda()
        
        adj_scale = 1.0
        return correlation_adj * adj_scale

    def forward_dynamic_gcn(self, x, dynamic_adj):
        x = torch.matmul(x, dynamic_adj)
        x = self.relu(x)
        x = self.dynamic_weight(x)
        x = self.relu(x)
        return x

    def forward(self, x):
        """
        Shape: 
        - Input: x(B, C_in, N) # C_in: 1024, N: num_classes
        - Output: (B, C_out, N) # C_out: 1024, N: num_classes
        """
        B, C, N = x.size() # B=b*n2*n1, C=c, N=h*w*2
        ## gcn ##
        dynamic_adj = self.forward_construct_correlation_graph(x) # (B, N, N)
        #print(dynamic_adj[0])
        out_dynamic = self.forward_dynamic_gcn(x, dynamic_adj) # (B, 1, N)

        ## gcn attention ##
        spatial_attention = self.attention_conv_1(out_dynamic).squeeze(1)
        spatial_attention_1 = spatial_attention[:,:N//2]
        spatial_attention_2 = spatial_attention[:,N//2:]
        # softmax_hw
        spatial_attention_1 = F.softmax(spatial_attention_1 / 0.025, dim=-1) + 1
        spatial_attention_2 = F.softmax(spatial_attention_2 / 0.025, dim=-1) + 1
        # sigmoid
        #spatial_attention_1 = torch.sigmoid(spatial_attention_1) + 1
        #spatial_attention_2 = torch.sigmoid(spatial_attention_2) + 1
        # attention op
        spatial_attention = torch.cat((spatial_attention_1, spatial_attention_2), 1)
        spatial_attention = spatial_attention.unsqueeze(1)# (B, 1, N)
        x = x * spatial_attention # spatial attention
        return x

