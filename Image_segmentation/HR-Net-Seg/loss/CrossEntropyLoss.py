import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label)

    def forward(self, pred, target):
        ph, pw = pred.size(2), pred.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            pred = F.interpolate(input=pred, size=(h, w), mode='bilinear', align_corners=False)

        loss = self.criterion(pred, target)
        return loss


