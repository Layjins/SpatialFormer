import torch
import torch.nn as nn
import torch.nn.functional as F


def redundancy_loss_func(f):
    """ 
    Shape: 
    - Input: (b, c, h, w) 
    - Output: (1)
    """
    # correlation
    b, c, h, w = f.size()
    f = f.contiguous().view(b, c, h*w)
    f_norm = F.normalize(f, p=2, dim=1, eps=1e-12) # (b, c, h*w)
    f_norm_t = f_norm.transpose(1, 2) # (b, h*w, c)
    correlation = torch.matmul(f_norm_t, f_norm) # (b, h*w, h*w)
    # target of redundancy matrix
    redundancy_param = 0.5 # default=0.0
    target = torch.eye(h*w)
    target[target==0] = redundancy_param
    target = target.unsqueeze(2).expand(h*w, h*w, b)
    target = target.permute(2,0,1) # (b, h*w, h*w)
    # mse loss
    mse_loss = nn.MSELoss()
    loss = mse_loss(correlation.cuda(), target.cuda())
    
    return loss

