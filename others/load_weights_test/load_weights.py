import torch
from model import resnet34
import os
import torch.nn as nn

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载权重
    # resnet34权重 url:https://download.pytorch.org/models/resnet34-333f7ec4.pth
    model_weights_path = './resnet34.pth'
    assert os.path.exists(model_weights_path), "model path:{} does not exists".format(model_weights_path)

    # 预定义网路结构
    net = resnet34()
    print(net)

    # 以下对象都为生成器，一个可迭代对象，使用for循环遍历其内容
    modules = net.modules()  # 遍历网络所有子层，子层指nn.Module子类。
    name_modules = net.named_modules()  # 不仅返回子层，也返回这些层的名字
    children = net.children()           # 遍历网络子层，即网络forward函数里面的层
    name_children = net.named_children()# 不仅返回子层，也返回这些层的名字，层的名字即forward函数里层的名字
    param = net.parameters()            # 遍历网络参数，池化层没有参数
    name_param = net.named_parameters() # 遍历网络参数，带名字
    # print(modules)           # <generator object Module.modules at 0x0000014D7D756580>

    module_list = [m for m in net.modules()]
    name_modules_list = [m for m in net.named_modules()]
    children_list = [m for m in net.children()]
    name_children_list = [m for m in net.named_children()]
    param_list = [m for m in net.parameters()]
    name_param_list = [m for m in net.named_parameters()]
    # print(module_list)

    # 模型权重的有序字典，可以通过修改state_dict来修改模型各层的参数，用于参数剪枝
    state_dict = net.state_dict()
    keys = net.state_dict().keys()
    values = net.state_dict().values()
    for key, value in net.state_dict().items():
        print("key: {} value: {}".format(key,value))
    # print(state_dict)

    # 对特定层进行处理（根据层名）
    for name, layer in net.named_modules():
        if "conv" in name:
            # do something
            print(name)
            # pass

    # 对特定层进行处理 （没有层名）
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d):
            # do something
            print(layer)
            # pass


    # 方法1：先加载预训练模型权重，赋值网络结构，再更改网络结构
    # 加载预训练权重参数
    # net = resnet34()
    pre_train_weights = torch.load(model_weights_path, map_location=device)
    # 载入网络结构
    net.load_state_dict(pre_train_weights, strict=True)  # strict=True:严格匹配网络结构
    # 改变全连接层结构
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 5)

    # 方法2： 先更改预训练模型权重，在将权值赋值给网络结构
    net = resnet34(num_classes=5)
    pre_train_weights = torch.load(model_weights_path, map_location=device)
    del_keys = []
    for key,_ in pre_train_weights.items():
        if "fc" in key:
            del_keys.append(key)

    for key in del_keys:
        del pre_train_weights[key]

    # strict=False: 网络结构和预训练权重不完全匹配
    # missing_keys：网络结构有某层结结构，但是权重模型中没有对应的层结构
    # unexpetected_keys：网络结构中没有某层结构，但是权重模型有对应层结构
    missing_keys, unexpected_keys = net.load_state_dict(pre_train_weights, strict=False)
    print("[missing_keys]:", *missing_keys, sep="\n")
    print("[unexpected_keys]:", *unexpected_keys, sep="\n")


if __name__ == '__main__':
    main()


