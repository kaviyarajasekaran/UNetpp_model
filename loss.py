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
        sobel_x = self.sobel_x.to(pred.dtype)
        sobel_y = self.sobel_y.to(pred.dtype)

        gx_pred = F.conv2d(pred, sobel_x, padding=1)
        gy_pred = F.conv2d(pred, sobel_y, padding=1)

        gx_target = F.conv2d(target, sobel_x, padding=1)
        gy_target = F.conv2d(target, sobel_y, padding=1)

        edge_pred = torch.sqrt(gx_pred**2 + gy_pred**2 + 1e-6)
        edge_target = torch.sqrt(gx_target**2 + gy_target**2 + 1e-6)

        return F.l1_loss(edge_pred, edge_target)


class L1_Edge_Loss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1 = nn.L1Loss()
        self.edge = EdgeLoss()

    def forward(self, pred, target):
        return self.alpha * self.l1(pred, target) + self.beta * self.edge(pred, target)
