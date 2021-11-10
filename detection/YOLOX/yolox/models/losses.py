#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import math

import torch
import torch.nn as nn


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)


        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)
        elif self.loss_type == 'alpha_iou':
            b1_x1, b1_x2 = pred[:, 0] - pred[:,2] / 2, pred[:, 0] + pred[:, 2] / 2
            b1_y1, b1_y2 = pred[:, 1] - pred[:,3] / 2, pred[:, 1] + pred[:, 3] / 2
            b2_x1, b2_x2 = target[:, 0] - target[0:, 2] / 2, target[:, 0] + target[0:, 2] / 2
            b2_y1, b2_y2 = target[:, 1] - target[0:, 3] / 2, target[:, 1] + target[0:, 3] / 2
            w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + 1e-16
            w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + 1e-16
            # w1, h1 = pred[:, 2] + 1e-16, pred[:, 3] + 1e-16
            # w2, h2 = target[:, 2]+1e-16, target[:, 3]+1e-16
            beta = 4
            cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
            ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
            c2 = cw ** beta + ch ** beta + 1e-16  # convex diagonal
            rho_x = torch.abs(b2_x1 + b2_x2 - b1_x1 - b1_x2)
            rho_y = torch.abs(b2_y1 + b2_y2 - b1_y1 - b1_y2)
            rho2 = (rho_x ** beta + rho_y ** beta) / (2 ** beta)
            v = (4/math.pi ** 2) *torch.pow(torch.atan(w2/h2) - torch.atan(w1/h1),2)
            with torch.no_grad():
                alpha_ciou = v / ((1 + 1e-16) - area_i / area_u + v)
                ciou = pow(iou,2) - (rho2 / c2 + torch.pow(v * alpha_ciou + 1e-16, 2))
                loss = 1.0 - ciou

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class FocalLoss(nn.Module):
    def __init__(self,gamma=2.0, alpha=0.25, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self,pred,label):
        # FL = - self.alpha * (1.0000001 - p_t) ** self.gamma * log(p_t)
        # log(p_t) = nn.BCEWithLogitsLoss()
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        """
        3样本，2分类 label=0或者1
        input = torch.randn(3,requires_grad=True)
        target = torch.FloatTensor([1,1,0])
        out = FocalLoss(input,target)
        """
        loss = self.loss_fcn(pred, label)
        pred_prob = torch.sigmoid(pred)
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss