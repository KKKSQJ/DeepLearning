import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.functional import pad

# 使用pytorch实现tf的卷积中的same
class conv2d(nn.Conv2d):

    def forward(self, input: Tensor) -> Tensor:
        # custom con2d, because pytorch don't have "padding='same'" option.
        def conv2d_same_padding(input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1):
            input_rows = input.size(2)
            filter_rows = weight.size(2)
            effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
            out_rows = (input_rows + stride[0] - 1) // stride[0]
            padding_needed = max(0, (out_rows - 1) * stride[0] + effective_filter_size_rows -
                                 input_rows)
            padding_rows = max(0, (out_rows - 1) * stride[0] +
                               (filter_rows - 1) * dilation[0] + 1 - input_rows)
            rows_odd = (padding_rows % 2 != 0)
            padding_cols = max(0, (out_rows - 1) * stride[0] +
                               (filter_rows - 1) * dilation[0] + 1 - input_rows)
            cols_odd = (padding_rows % 2 != 0)

            if rows_odd or cols_odd:
                input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

            return F.conv2d(input, weight, bias, stride,
                            padding=(padding_rows // 2, padding_cols // 2),
                            dilation=dilation, groups=groups)

        return conv2d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)


