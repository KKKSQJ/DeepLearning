// my_add.cpp

#include<torch/torch.h>

torch::Tensor my_add(torch::Tensor a, torch::Tensor b) {
	return 3 * a + 2 * b;
}

// 模块名：my_lib 可作为python的导入包
// 双引导中的“my_add”：可作为python导入包下的函数被python脚本调用
PYBIND11_MODULE(my_lib, m) {
	m.def("my_add", my_add);
}