import torch
import torch.nn as nn
import numpy as np

# feature1 = np.array([[[0, 1], [1, -1]], [[-1, 0], [0, 2]]])
# feature2 = np.array([[[1, 0], [3, 1]], [[0, 1], [4, -1]]])
"""
    Examples::

        >>> input = torch.randn(20, 6, 10, 10)
        >>> # Separate 6 channels into 3 groups
        >>> m = nn.GroupNorm(3, 6)
        >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
        >>> m = nn.GroupNorm(6, 6)
        >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
        >>> m = nn.GroupNorm(1, 6)
        >>> # Activating the module
        >>> output = m(input)
"""
sample1 = torch.tensor([[[1, 1], [1, 2]], [[-1, 1], [0, 1]]], dtype=torch.float32)
sample2 = torch.tensor([[[0, -1], [2, 2]], [[0, -1], [3, 1]]], dtype=torch.float32)
gn = nn.GroupNorm(2, 2)
output1 = gn(sample1.unsqueeze(0))
output2 = gn(sample2.unsqueeze(0))
print(output1)
# print(output2)

IN = nn.InstanceNorm2d(2)
in_output1 = IN(sample1.unsqueeze(0))
in_output2 = IN(sample2.unsqueeze(0))

mean = np.mean(sample1.numpy(), axis=(1, 2))
var = np.var(sample1.numpy(), axis=(1, 2))
output3 = (sample1.numpy() - mean[:, None, None]) / np.sqrt(var[:, None, None] + 1e-5)
print(mean)
print(var)
print(output3)
