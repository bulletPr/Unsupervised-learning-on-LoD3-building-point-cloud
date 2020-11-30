
#
#
#      0=================================0
#      |    Project Name                 |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Implements: Calculate ChamferLoss
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      YUWEI CAO - 2020/10/22 09:32 AM 
#
#


# ----------------------------------------
# import packages
# ----------------------------------------

import torch
import torch.nn as nn
#from torch.autograd import Variable
import torch.nn.functional as F


# ----------------------------------------
# ChamferLoss
# ----------------------------------------

class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x)
        diag_ind_y = torch.arange(0, num_points_y)
        if x.get_device() != -1:
            diag_ind_x = diag_ind_x.cuda(x.get_device())
            diag_ind_y = diag_ind_y.cuda(x.get_device())
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)
        return loss_1 + loss_2



class ChamferLoss_m(nn.Module):
    def __init__(self):
        super(ChamferLoss_m, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def ChamferDistance(self, x, y):  # for example, x = batch,2025,3 y = batch,2048,3
        #   compute chamfer distance between tow point clouds x and y

        x_size = x.size()
        y_size = y.size()
        assert (x_size[0] == y_size[0])
        assert (x_size[2] == y_size[2])
        x = torch.unsqueeze(x, 1)  # x = batch,1,2025,3
        y = torch.unsqueeze(y, 2)  # y = batch,2048,1,3

        x = x.repeat(1, y_size[1], 1, 1)  # x = batch,2048,2025,3
        y = y.repeat(1, 1, x_size[1], 1)  # y = batch,2048,2025,3

        x_y = x - y
        x_y = torch.pow(x_y, 2)  # x_y = batch,2048,2025,3
        x_y = torch.sum(x_y, 3, keepdim=True)  # x_y = batch,2048,2025,1
        x_y = torch.squeeze(x_y, 3)  # x_y = batch,2048,2025
        x_y_row, _ = torch.min(x_y, 1, keepdim=True)  # x_y_row = batch,1,2025
        x_y_col, _ = torch.min(x_y, 2, keepdim=True)  # x_y_col = batch,2048,1

        x_y_row = torch.mean(x_y_row, 2, keepdim=True)  # x_y_row = batch,1,1
        x_y_col = torch.mean(x_y_col, 1, keepdim=True)  # batch,1,1
        x_y_row_col = torch.cat((x_y_row, x_y_col), 2)  # batch,1,2
        chamfer_distance, _ = torch.max(x_y_row_col, 2, keepdim=True)  # batch,1,1
        # chamfer_distance = torch.reshape(chamfer_distance,(x_size[0],-1))  #batch,1
        # chamfer_distance = torch.squeeze(chamfer_distance,1)    # batch
        chamfer_distance = torch.mean(chamfer_distance)
        return chamfer_distance


    def forward(self, preds, gts):
        return self.ChamferDistance(preds, gts)


# ----------------------------------------
# CrossEntropyLoss
# ----------------------------------------

class CrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=True):
        super(CrossEntropyLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, preds, gts):
        gts = gts.contiguous().view(-1)

        if self.smoothing:
            eps = 0.2
            n_class = preds.size(1)

            one_hot = torch.zeros_like(preds).scatter(1, gts.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(preds, dim=1)

            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = F.cross_entropy(preds, gts, reduction='mean')

        return loss
