import torch
import numpy as np
import torch.nn.functional as F


class KpLoss(object):
    def __init__(self):
        self.criterion = torch.nn.MSELoss(reduction='none')

    def __call__(self, logits, targets):
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        device = logits.device
        bs = logits.shape[0]
        # [num_kps, H, W] -> [B, num_kps, H, W]
        heatmaps = torch.stack([t["heatmap"].to(device) for t in targets])
        # [num_kps] -> [B, num_kps]
        # kps_weights = torch.stack([t["kps_weights"].to(device) for t in targets])
        kps_weights = torch.ones((bs, heatmaps.shape[1])).to(heatmaps.device)

        # [B, num_kps, H, W] -> [B, num_kps]
        loss = self.criterion(logits, heatmaps).mean(dim=[2, 3])
        loss = torch.sum(loss * kps_weights) / bs
        # return loss
        return loss


# focal mse
class Kploss_focal(object):
    def __init__(self, pos_neg_weights=10, gamma=2):
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.pos_neg_weights = pos_neg_weights
        self.gamma = gamma

    def __call__(self, logits, targets):
        assert len(logits.shape) == 4, 'logits should be 4-ndim'
        device = logits.device
        bs, c, h, w = logits.shape
        # [num_kps, H, W] -> [B, num_kps, H, W]
        heatmaps = torch.stack([t["heatmap"].to(device) for t in targets])
        pos_index = torch.where(heatmaps != 0)
        neg_index = torch.where(heatmaps == 0)

        # [num_kps] -> [B, num_kps]
        # kps_weights = torch.stack([t["kps_weights"].to(device) for t in targets])
        kps_weights = torch.ones((bs, heatmaps.shape[1])).to(heatmaps.device)

        # [B, num_kps, H, W] -> [B, num_kps]
        # loss = self.criterion(logits, heatmaps).mean(dim=[2, 3])
        loss = self.criterion(logits, heatmaps)

        # 样本的难易程度
        loss = torch.pow(loss, self.gamma)

        # 正负样本平衡
        loss[pos_index] = loss[pos_index] * self.pos_neg_weights

        loss = loss.mean(dim=[2, 3])
        loss = torch.sum(loss * kps_weights) / bs

        # return loss
        return loss


def _reg_loss(regs, gt_regs, mask):
    n = sum([len(torch.where(m == 1)[0]) for m in mask])
    mask = mask[:, :, None].expand_as(gt_regs).float()
    loss = sum(F.l1_loss(r * mask, gt_regs * mask, reduction='sum') / (n + 1e-4) for r in regs)
    # loss = sum(F.l1_loss(r * mask, gt_regs * mask, reduction='sum') / (mask.sum() + 1e-4) for r in regs)
    # return loss / len(regs)
    return loss


def _neg_loss(preds, targets):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        preds (B x c x h x w)
        gt_regr (B x c x h x w)
    '''
    pos_inds = targets.eq(1).float()
    neg_inds = targets.lt(1).float()

    neg_weights = torch.pow(1 - targets, 4)

    loss = 0
    for pred in preds:
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred,
                                                   2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss / len(preds)


def _neg_loss_slow(preds, targets):
    pos_inds = targets == 1  # todo targets > 1-epsilon ?
    neg_inds = targets < 1  # todo targets < 1-epsilon ?

    neg_weights = torch.pow(1 - targets[neg_inds], 4)

    loss = 0
    for pred in preds:
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * \
                   torch.pow(neg_pred, 2) * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss
