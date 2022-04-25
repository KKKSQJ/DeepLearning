import torch.nn as nn
from losses.SupConLoss import SupConLoss
from losses.LabelSmooth import LabelSmoothingLoss

LOSSES = {'SupCon': SupConLoss,
          'LabelSmoothing': LabelSmoothingLoss,
          'CrossEntropy': nn.CrossEntropyLoss}
