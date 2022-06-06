import torch
import torch.nn as nn
import numpy as np

# feature1 = np.array([[[1, 1], [1, 2]], [[-1, 1], [0, 1]]])
# feature2 = np.array([[[0, -1], [2, 2]], [[0, -1], [3, 1]]])
"""
    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm2d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)
"""

feature1 = torch.tensor([[[1, 1], [1, 2]], [[-1, 1], [0, 1]]], dtype=torch.float32)
feature2 = torch.tensor([[[0, -1], [2, 2]], [[0, -1], [3, 1]]], dtype=torch.float32)
input = torch.stack([feature1, feature2], dim=0)
bn = nn.BatchNorm2d(2)
output = bn(input)
print(output)


