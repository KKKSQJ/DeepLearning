"""
MSE & mask loss for mae Unsupervised training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def build_mask(mask_index, patch_size, img_size):
    num_pathces = img_size // patch_size
    mask_map = torch.zeros((img_size, img_size)).float()
    # reshape the h w -> n c
    mask_map = rearrange(mask_map, '(h p1) (w p2) -> (h w) (p1 p2)', h=num_pathces, w=num_pathces, p1=patch_size,
                         p2=patch_size)
    mask_index = [index - 1 for index in mask_index]
    mask_map[mask_index] = 1.
    # reshape the n c -> h w
    mask_map = rearrange(mask_map, '(h w) (p1 p2) -> (h p1) (w p2)', h=num_pathces, w=num_pathces, p1=patch_size,
                         p2=patch_size)
    return mask_map


# class MSELoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, pred, target, mask_map):
#         # print(pred.shape)
#         # print(target.shape)
#         # print(mask_map.shape)
#         pred = pred * mask_map.cuda()
#         target = target * mask_map.cuda()
#         loss = F.mse_loss(pred, target)
#         return loss


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, pred, target):
        loss = F.mse_loss(pred, target)
        return loss


if __name__ == '__main__':
    mask_index = [1, 2, 3, 4, 5, 6]
    patch_size = 16
    img_size = 224
    loss = MSELoss(mask_index, patch_size, img_size)
    # print(loss)
    pred = torch.randn(1, 3, 224, 224)
    target = torch.randn(1, 3, 224, 224)
    losses = loss(pred, target)
    print(losses)
