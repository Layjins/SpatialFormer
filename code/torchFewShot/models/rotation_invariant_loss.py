import torch
import torch.nn as nn
import torch.nn.functional as F


def rotation_invariant_loss_func(f, shape_param):
    """ 
    Shape: 
    - Input: (b, c, h, w) 
    - Output: (1)
    """
    bs, n, c, h, w = shape_param
    # split to 4 groups by angles
    b, c, h, w = f.size() # b = bs*n
    f = f.contiguous().view(bs, n, c, h, w)
    f = f.contiguous().view(bs, n//4, 4, c, h, w)
    f = f.contiguous().view(bs*n//4, 4, c, h, w)
    f = f.contiguous().view(bs*n//4, 4, c, h*w)
    f_angle0 = f[:,0,:,:]
    f_angle90 = f[:,1,:,:]
    f_angle180 = f[:,2,:,:]
    f_angle270 = f[:,3,:,:]
    # correlation
    f_angle0_norm = F.normalize(f_angle0, p=2, dim=1, eps=1e-12) # (bs*n//4, c, h*w)
    f_angle90_norm = F.normalize(f_angle90, p=2, dim=1, eps=1e-12) # (bs*n//4, c, h*w)
    f_angle180_norm = F.normalize(f_angle180, p=2, dim=1, eps=1e-12) # (bs*n//4, c, h*w)
    f_angle270_norm = F.normalize(f_angle270, p=2, dim=1, eps=1e-12) # (bs*n//4, c, h*w)
    f_angle0_norm_t = f_angle0_norm.transpose(1, 2) # (bs*n//4, h*w, c)
    correlation_angle90 = torch.matmul(f_angle0_norm_t, f_angle90_norm) # (bs*n//4, h*w, h*w)
    correlation_angle180 = torch.matmul(f_angle0_norm_t, f_angle180_norm) # (bs*n//4, h*w, h*w)
    correlation_angle270 = torch.matmul(f_angle0_norm_t, f_angle270_norm) # (bs*n//4, h*w, h*w)
    # softmax
    #correlation_angle90 = F.softmax(correlation_angle90, dim=-1)
    #correlation_angle180 = F.softmax(correlation_angle180, dim=-1)
    #correlation_angle270 = F.softmax(correlation_angle270, dim=-1)
    # sigmoid
    correlation_angle90 = F.sigmoid(correlation_angle90)
    correlation_angle180 = F.sigmoid(correlation_angle180)
    correlation_angle270 = F.sigmoid(correlation_angle270)

    # target of invariant and redundancy matrix
    invariant_param = 1.0 # default=1.0
    redundancy_param = 0.0 # default=0.0
    target = torch.eye(h*w)
    target[target==0] = redundancy_param
    target[target==1] = invariant_param
    target = target.unsqueeze(2).expand(h*w, h*w, bs*n//4)
    tartet = target.permute(2,0,1) # (bs*n//4, h*w, h*w)
    # mse loss
    #mse_loss = nn.MSELoss(size_average = False)
    mse_loss = nn.MSELoss()
    loss_angle90 = mse_loss(correlation_angle90.cuda(), tartet.cuda())
    loss_angle180 = mse_loss(correlation_angle180.cuda(), tartet.cuda())
    loss_angle270 = mse_loss(correlation_angle270.cuda(), tartet.cuda())
    loss_scale = 1 # default=1
    #loss = loss_scale * (loss_angle90 + loss_angle180 + loss_angle270) / (bs*n//4) / 3.0 # default=3.0
    loss = (loss_angle90 + loss_angle180 + loss_angle270) / 3.0 # default=3.0
    #print(loss)
    return loss

