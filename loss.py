import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()

        sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).view(1,1,3,3)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def forward(self, pred, target):
        gx_pred = F.conv2d(pred, self.sobel_x, padding=1)
        gy_pred = F.conv2d(pred, self.sobel_y, padding=1)

        gx_target = F.conv2d(target, self.sobel_x, padding=1)
        gy_target = F.conv2d(target, self.sobel_y, padding=1)

        return F.l1_loss(gx_pred, gx_target) + F.l1_loss(gy_pred, gy_target)


class L1_Edge_Loss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.edge = EdgeLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        return self.alpha * self.l1(pred, target) + self.beta * self.edge(pred, target)
