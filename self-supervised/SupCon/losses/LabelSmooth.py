import torch.nn as nn
import torch


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0., dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


if __name__ == '__main__':
    loss = LabelSmoothingLoss(classes=5, smoothing=0.1)
    inputs = torch.randn(2, 5)
    targets = torch.empty(2, dtype=torch.long).random_(5)
    out = loss(inputs,targets)
    print(out)
