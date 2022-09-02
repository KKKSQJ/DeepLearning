"""
测试pytorch模型到onnx的算子转换

3. 测试支持 自定义 全新算子，为Pytorch添加C++扩展

我们已知

需要完成
pytorch新算子实现
ONNX新算子实现
算子从 Aten到ONNX的符号连接

因此，为算子添加符号函数一般要经过以下几步：
1.获取原算子的前向推理接口。
2.获取目标 ONNX 算子的定义。
3.编写符号函数并绑定。
"""

# 假设我们定义了一个全新算子，该算子的输入是张量a,b，输出3*a+2*b

# 第一步：为Pytorch添加C++扩展
"""

1.新建一个“my_add.cpp”
2. 

// my_add.cpp 
 
#include <torch/torch.h> 
 
torch::Tensor my_add(torch::Tensor a, torch::Tensor b) 
{ 
    return 3 * a + 2* b; 
} 

// PYBIND11_MODULE:为C++函数提供python调用接口。这里my_lib是未来在Python里导入的模块名
// 双引号中的 “my_add”是python调用的接口名称。这里是与c++函数的名称对齐

PYBIND11_MODULE(my_lib, m) 
{ 
    m.def("my_add", my_add); 
} 
"""

# 第二步：编写setup.py ，来编译刚刚的c++文件
"""

1.新建 “setup.py”
2.

# 使用setuotools的编译功能
from setuptools import setup
# 使用Pytorch中C++扩展根据函数 
from torch.utils import cpp_extension 
 
setup(
      # name:生成的对外的接口名字
      name='my_add', 
      # my_lib:模块名字 my_add.cpp：模块对应的源文件
      ext_modules=[cpp_extension.CppExtension('my_lib', ['my_add.cpp'])], 
      cmdclass={'build_ext': cpp_extension.BuildExtension}
      ) 

3.在命令行，执行 python setup.py 自动编译C++代码
"""

# 第三步：用torch.autograd.Function 进行封装
import torch
import my_lib

a = my_lib.my_add(torch.Tensor([1]), torch.Tensor([2]))
print(a)


class MyAddFunction(torch.autograd.Function):
    """
    在 forward 函数中，我们用 my_lib.my_add(a, b) 就可以调用之前写的C++函数了。
    这里 my_lib 是库名，my_add 是函数名，这两个名字是在前面C++的 PYBIND11_MODULE 中定义的。
    """

    @staticmethod
    def forward(ctx, a, b):
        return my_lib.my_add(a, b)

    """
    定义了 symbolic 静态方法，该 Function 在执行 torch.onnx.export() 时就可以根据 symbolic 中定义的规则转换成 ONNX 算子。
    这个 symbolic 就是前面提到的符号函数，只是它的名称必须是 symbolic 而已。
    """

    @staticmethod
    def symbolic(g, a, b):
        """
        在 symbolic 函数中，我们用 g.op() 定义了三个算子：常量、乘法、加法。
        这里乘法和加法的用法和前面提到的 asinh 一样，只需要根据 ONNX 算子定义规则把输入参数填入即可。
        而在定义常量算子时，我们要把 PyTorch 张量的值传入 value_t 参数中。
        """
        three = g.op("Constant", value_t=torch.tensor([3]))
        two = g.op("Constant", value_t=torch.tensor([2]))

        """
        在 ONNX 中，我们需要把新建常量当成一个算子来看待，尽管这个算子并不会以节点的形式出现在 ONNX 模型的可视化结果里。
        """
        a = g.op("Mul", a, three)
        b = g.op("Mul", b, two)
        return g.op("Add", a, b)


# 第四步：把算子封装成 Function 后，我们可以把 my_add算子用起来了。

"""
apply是torch.autograd.Function 的一个方法，这个方法完成了 Function 在前向推理或者反向传播时的调度。
我们在使用 Function 的派生类做推理时，不应该显式地调用 forward()，而应该调用其 apply 方法。

这里我们使用 my_add = MyAddFunction.apply 把这个调用方法取了一个更简短的别名 my_add。
以后在使用 my_add 算子时，我们应该忽略 MyAddFunction 的实现细节，而只通过 my_add 这个接口来访问算子。
这里 my_add 的地位，和 PyTorch 的 asinh, interpolate, conv2d等原生函数是类似的。
"""
my_add = MyAddFunction.apply


class MyAdd(torch.nn.Module):
    def __init__(self):
        super(MyAdd, self).__init__()

    def forward(self, a, b):
        # 这里的my_add就和asinh、conv2d等原生函数一样。my_add完成了对MyAddFunction的封装，我们不管MyAddFunction的具体实现，只需要正确调用即可
        return my_add(a, b)


# 第五步：测试算子
model = MyAdd()
input = torch.rand(1, 3, 10, 10)
# (input,input)是因为，网络的输入是两个参数
torch.onnx.export(model, (input, input), "my_add.onnx")
print("export succeed~")
torch_output = model(input, input).detach().numpy()

import onnxruntime
import numpy as np

sess = onnxruntime.InferenceSession("my_add.onnx")
ort_output = sess.run(None, {'a': input.numpy(), 'b': input.numpy()})[0]

assert np.allclose(torch_output, ort_output)
print('OK')
