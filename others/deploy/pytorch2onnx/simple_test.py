import torch
import torchvision
import numpy as np


def pt_to_onnx():
    # input
    dummy_input = torch.randn(10, 3, 224, 224, device="cuda")
    model = torchvision.models.alexnet(pretrained=False)
    model.load_state_dict(torch.load("alexnet-owt-7be5be79.pth", map_location='cpu'), strict=True)
    model.cuda()

    input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]
    output_names = ["output1"]

    # pt模型转ONNX
    torch.onnx.export(
        model,  # 需要转换的模型，支持的模型类型有：torch.nn.Module, torch.jit.ScriptModule or torch.jit.ScriptFunction
        dummy_input,  # 输入的参数，可以是3中形式。1.一个tuple。2.一个Tensor。3.一个带有字典的tuple。Tensor类型的参数会被当做onnx模型的输入
        "alexnet.onnx",  # 一个文件类对象或一个路径字符串，二进制的protocol buffer将被写入此文件
        export_params=True,  # 如果为True则导出模型的参数。如果想导出一个未训练的模型，则设为False
        verbose=True,  # 如果为True，则打印一些转换日志，并且onnx模型中会包含doc_string信息。
        input_names=input_names,  # (list of str, default empty list) 按顺序分配给onnx图的输入节点的名称列表。
        output_names=output_names,  # (list of str, default empty list) 按顺序分配给onnx图的输出节点的名称列表。
        # training=TrainingMode.EVA # (enum, default TrainingMode.EVAL) 以推理模式导出模型。
        # opset_version (int, default 9) # 默认是9。值必须等于_onnx_main_opset或在_onnx_stable_opsets之内。具体可在torch/onnx/symbolic_helper.py中找到
        # dynamic_axes (dict<string, dict<python:int, string>> or dict<string, list(int)>, default empty dict) KEY(str) - 必须是input_names或output_names指定的名称，用来指定哪个变量需要使用到动态尺寸。

    )


def eval_onnx():
    # onnx模型验证
    import onnx

    # Load the ONNX model
    model = onnx.load("alexnet.onnx")

    # Check that the model is well formed
    onnx.checker.check_model(model)

    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(model.graph))


def eval_onnxruntime():
    # onnx模型验证
    import onnxruntime as ort

    ort_session = ort.InferenceSession("alexnet.onnx")

    outputs = ort_session.run(None, {"actual_input_1": np.random.randn(10, 3, 224, 224).astype(np.float32)})

    print(outputs[0])
    print(outputs[0].shape)


if __name__ == '__main__':
    # pt模型转onnx
    # pt_to_onnx()

    # onnx验证
    eval_onnxruntime()
