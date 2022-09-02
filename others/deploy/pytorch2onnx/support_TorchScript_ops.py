"""
测试pytorch模型到onnx的算子转换

2. 测试支持 TorchScript 算子，自定义ONNX算子

我们已知
Aten算子的接口定义（pytorch算子）

需要完成
ONNX算子的自定义
算子从 Aten到ONNX的符号连接即可

因此，为算子添加符号函数一般要经过以下几步：
1.获取原算子的前向推理接口。
2.获取目标 ONNX 算子的定义。
3.编写符号函数并绑定。
"""
import onnxruntime
import torch
import torchvision
import numpy as np


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 18, 3)
        self.conv2 = torchvision.ops.DeformConv2d(3, 3, 3)

    def forward(self, x):
        return self.conv2(x, self.conv1(x))


from torch.onnx import register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args


@parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "none")
def symbolic(g,
             input,
             weight,
             offset,
             mask,
             bias,
             stride_h, stride_w,
             pad_h, pad_w,
             dil_h, dil_w,
             n_weight_grps,
             n_offset_grps,
             use_mask):
    return g.op("custom::deform_conv2d", input, offset)


register_custom_op_symbolic("torchvision::deform_conv2d", symbolic, 9)

model = Model()
input = torch.rand(1, 3, 10, 10)
torch.onnx.export(model, input, "dcn.onnx")
print("export succeed~")


def test():
    model = Model()
    input = torch.randn(1, 3, 10, 10)
    torch_output = model(input).detach().numpy()

    sess = onnxruntime.InferenceSession("dcn.onnx")
    ort_output = sess.run(None, {"0": input.numpy()})[0]

    # 使用 np.allclose 来保证两个结果张量的误差在一个可以允许的范围内。一切正常的话，运行这段代码后，assert 所在行不会报错，程序应该没有任何输出。
    assert np.allclose(torch_output, ort_output)
    print("test succeed!")

# if __name__ == '__main__':
#     test()
