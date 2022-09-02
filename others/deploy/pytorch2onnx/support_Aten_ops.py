
"""
测试pytorch模型到onnx的算子转换

1. 测试支持Aten算子

我们已知
Aten算子的接口定义（pytorch算子）
ONNX对应算子的实现

需要完成算子从 Aten到ONNX的符号连接即可

因此，为算子添加符号函数一般要经过以下几步：
1.获取原算子的前向推理接口。
2.获取目标 ONNX 算子的定义。
3.编写符号函数并绑定。
"""

import torch
import onnxruntime
import torch
import numpy as np


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        return torch.asinh(x)


from torch.onnx.symbolic_registry import register_op


# asinh在pytorch中的算子接口定义（Aten算子）
# def asinh(input: Tensor, *, out: Optional[Tensor]=None) -> Tensor: ...
def asinh_symbolic(g, input, *, out=None):  # 除g外的参数必须与Aten算子接口定义保持一致
    # 调用g.op函数，其中"Asinh"是ONNX算子名字，input则是onnx算子参数。注：asinh只有一个参数input
    return g.op("Asinh", input)

    # 将Aten算子和符号函数绑定在一起，则完成关系映射。


# def register_op(opname, op, domain, version)
register_op('asinh', asinh_symbolic, '', 9)


def export():
    model = Model()
    input = torch.randn(1, 3, 10, 10)
    torch.onnx.export(model, input, "asinh.onnx")
    print("export succeed~")


def test():
    model = Model()
    input = torch.randn(1, 3, 10, 10)
    torch_output = model(input).detach().numpy()

    sess = onnxruntime.InferenceSession('asinh.onnx')
    # onnx的输入是numpy格式：BCHW
    ort_output = sess.run(None, {'0': input.numpy()})[0]

    # 使用 np.allclose 来保证两个结果张量的误差在一个可以允许的范围内。一切正常的话，运行这段代码后，assert 所在行不会报错，程序应该没有任何输出。
    assert np.allclose(torch_output, ort_output)
    print("test succeed!")


if __name__ == '__main__':
    export()
    test()

    # 测试DCN


