import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

class EMD_module(nn.Module):
    def __init__(self):
        super(EMD_module, self).__init__()
        self.args_norm = 'center' # choices=['center', 'none']
        self.args_metric = 'cosine' # choices=['cosine', 'l2']
        self.args_temperature = 12.5 # default=12.5

    def forward(self, in_support, in_query):
        # in_support = b, n2, n1, c, h, w 
        # in_query = b, n2, n1, c, h, w 
        b, n2, n1, c, h, w = in_support.size()
        in_support = in_support.view(b*n2*n1, c, h, w)
        in_query = in_query.view(b*n2*n1, c, h, w)
        logits = torch.zeros(b*n2*n1).cuda()
        for i in range(b*n2*n1):
            support = in_support[i].unsqueeze(0)
            query = in_query[i].unsqueeze(0)
            # support = n, c, h, w 
            # query = n, c, h, w 
            weight_1 = self.get_weight_vector(query, support)
            weight_2 = self.get_weight_vector(support, query)

            support = self.normalize_feature(support)
            query = self.normalize_feature(query)

            similarity_map = self.get_similiarity_map(support, query)
            logit = self.get_emd_distance(similarity_map, weight_1, weight_2)
            logits[i] = logit
        return logits

    def get_weight_vector(self, A, B):

        M = A.shape[0]
        N = B.shape[0]

        B = F.adaptive_avg_pool2d(B, [1, 1])
        B = B.repeat(1, 1, A.shape[2], A.shape[3])

        A = A.unsqueeze(1)
        B = B.unsqueeze(0)

        A = A.repeat(1, N, 1, 1, 1)
        B = B.repeat(M, 1, 1, 1, 1)

        combination = (A * B).sum(2)
        combination = combination.view(M, N, -1)
        combination = F.relu(combination) + 1e-3
        return combination

    def normalize_feature(self, x):
        if self.args_norm == 'center':
            x = x - x.mean(1).unsqueeze(1)
            return x
        else:
            return x

    def get_similiarity_map(self, proto, query):
        way = proto.shape[0]
        num_query = query.shape[0]
        query = query.view(query.shape[0], query.shape[1], -1)
        proto = proto.view(proto.shape[0], proto.shape[1], -1)

        proto = proto.unsqueeze(0).repeat([num_query, 1, 1, 1])
        query = query.unsqueeze(1).repeat([1, way, 1, 1])
        proto = proto.permute(0, 1, 3, 2)
        query = query.permute(0, 1, 3, 2)
        feature_size = proto.shape[-2]

        if self.args_metric == 'cosine':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = F.cosine_similarity(proto, query, dim=-1)
        if self.args_metric == 'l2':
            proto = proto.unsqueeze(-3)
            query = query.unsqueeze(-2)
            query = query.repeat(1, 1, 1, feature_size, 1)
            similarity_map = (proto - query).pow(2).sum(-1)
            similarity_map = 1 - similarity_map
        return similarity_map

    def emd_inference_opencv(self, cost_matrix, weight1, weight2):
        # cost matrix is a tensor of shape [N,N]
        cost_matrix = cost_matrix.detach().cpu().numpy()

        weight1 = F.relu(weight1) + 1e-5
        weight2 = F.relu(weight2) + 1e-5

        weight1 = (weight1 * (weight1.shape[0] / weight1.sum().item())).view(-1, 1).detach().cpu().numpy()
        weight2 = (weight2 * (weight2.shape[0] / weight2.sum().item())).view(-1, 1).detach().cpu().numpy()

        cost, _, flow = cv2.EMD(weight1, weight2, cv2.DIST_USER, cost_matrix)
        return cost, flow

    def get_emd_distance(self, similarity_map, weight_1, weight_2):
        num_query = similarity_map.shape[0]
        num_proto = similarity_map.shape[1]
        num_node=weight_1.shape[-1]

        for i in range(num_query):
            for j in range(num_proto):
                _, flow = self.emd_inference_opencv(1 - similarity_map[i, j, :, :], weight_1[i, j, :], weight_2[j, i, :])
                similarity_map[i, j, :, :] =(similarity_map[i, j, :, :])*torch.from_numpy(flow).cuda()

        temperature=(self.args_temperature/num_node)
        logitis = similarity_map.sum(-1).sum(-1) *  temperature
        return logitis

