"""CenterLoss"""
import copy
import torch as t
import torch.nn as nn
import torch.nn.functional as F


class CenterLoss(nn.Module):
    def __init__(self, cls_num, featur_num, args):
        super().__init__()

        self.cls_num = cls_num
        self.featur_num = featur_num
        self.center = nn.Parameter(t.rand(cls_num, featur_num)).cuda(args.local_rank)  # 全部类别

    def forward(self, xs, ys):  # xs=feature, ys=target
        xs = F.normalize(xs)
        y = copy.copy(ys)
        if len(y.shape) == 2:
            y = t.argmax(y, dim=1)
        self.center_exp = self.center.index_select(dim=0, index=y.long())
        count = t.histc(ys, bins=self.cls_num, min=0, max=self.cls_num - 1)
        self.count_dis = count.index_select(dim=0, index=y.long()) + 1
        loss = t.sum(t.sum((xs - self.center_exp) ** 2, dim=1) / 2.0 / self.count_dis.float())
        # print("center:", self.center)

        return loss


if __name__ == "__main__":
    loss = CenterLoss(3, 32)
    inputs = t.randn(8, 32)
    labels = t.tensor([0, 0, 0, 1, 2, 1, 2, 1]).float()
    losses = loss(inputs, labels)
    print(losses)
