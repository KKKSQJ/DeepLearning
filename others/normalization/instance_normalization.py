import torch
import torch.nn as nn
import numpy as np

# feature1 = np.array([[[0, 1], [1, -1]], [[-1, 0], [0, 2]]])
# feature2 = np.array([[[1, 0], [3, 1]], [[0, 1], [4, -1]]])
"""
    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples::

        >>> # Without Learnable Parameters
        >>> m = nn.InstanceNorm2d(100)
        >>> # With Learnable Parameters
        >>> m = nn.InstanceNorm2d(100, affine=True)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)
"""
sample1 = torch.tensor([[[1, 1], [1, 2]], [[-1, 1], [0, 1]]],dtype=torch.float32)
sample2 = torch.tensor([[[0, -1], [2, 2]], [[0, -1], [3, 1]]],dtype=torch.float32)
IN = nn.InstanceNorm2d(2)
output1 = IN(sample1.unsqueeze(0))
output2 = IN(sample2.unsqueeze(0))
print(output1)
# print(output2)



mean = np.mean(sample1.numpy(),axis=(1,2))
var = np.var(sample1.numpy(),axis=(1,2))
output3 = (sample1.numpy()-mean[:,None,None])/np.sqrt(var[:,None,None]+1e-5)
print(mean)
print(var)
print(output3)

