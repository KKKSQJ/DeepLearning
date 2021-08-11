import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

input = torch.randn(2, 3)
# input = input.view(-1,1)
print(input)
target = torch.empty(2, dtype=torch.long).random_(3)
print(target)
target = target.view(-1,1)
print("target:{}".format(target))
logpt = F.log_softmax(input)
print("log:{}".format(logpt))
logpt = logpt.gather(1, target)
print(logpt)
logpt = logpt.view(-1)
print(logpt)
pt = Variable(logpt.data.exp())
print(pt)