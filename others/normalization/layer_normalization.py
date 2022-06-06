import torch
import torch.nn as nn
import numpy as np

# feature1 = np.array([[[0, 1], [1, -1]], [[-1, 0], [0, 2]]])
# feature2 = np.array([[[1, 0], [3, 1]], [[0, 1], [4, -1]]])
"""
    Examples::

        >>> input = torch.randn(20, 5, 10, 10)
        >>> # With Learnable Parameters
        >>> m = nn.LayerNorm(input.size()[1:])
        >>> # Without Learnable Parameters
        >>> m = nn.LayerNorm(input.size()[1:], elementwise_affine=False)
        >>> # Normalize over last two dimensions
        >>> m = nn.LayerNorm([10, 10])
        >>> # Normalize over last dimension of size 10
        >>> m = nn.LayerNorm(10)
        >>> # Activating the module
        >>> output = m(input)
"""
sample1 = torch.tensor([[[1, 1], [1, 2]], [[-1, 1], [0, 1]]],dtype=torch.float32)
sample2 = torch.tensor([[[0, -1], [2, 2]], [[0, -1], [3, 1]]],dtype=torch.float32)
ln = nn.LayerNorm([2,2,2],elementwise_affine=False)
output1 = ln(sample1)
output2 = ln(sample2)
print(output1)
# print(output2)

mean = np.mean(sample1.numpy())
var = np.var(sample1.numpy())
output3 = (sample1.numpy()-mean)/np.sqrt(var+1e-5)
print(mean)
print(var)
print(output3)

