# 目录
- pytorch2onnx：用于将pytorch模型转换为onnx。自定义算子
- onnx2ncnn
- onn2tensorRT

## pytorch2onnx
### 参数讲解
核心代码：```torch.onnx.export```

```
def export(
           model,  # pytorch模型 (torch.nn.Module, torch.jit.ScriptModule or torch.jit.ScriptFunction)
           args,   # 模型输入 args = (x, y, z) 或者 args = torch.Tensor([1, 2, 3]) 
           f,      # 模型输出 xxx.onnx 一个文件类对象或一个路径字符串，二进制的protocol buffer将被写入此文件
           export_params=True,  # 是否存储模型权重，如果为True则导出模型的参数。如果想导出一个未训练的模型，则设为False
           verbose=False,       # 如果为True，则打印一些转换日志，并且onnx模型中会包含doc_string信息。
           training=TrainingMode.EVAL, # (enum, default TrainingMode.EVAL)
           input_names=None,  # 按顺序分配给onnx图的输入节点的名称列表。
           output_names=None, # 按顺序分配给onnx图的输出节点的名称列表。
           aten=False, 
           export_raw_ir=False, 
           operator_export_type=None, 
           opset_version=None, # 算子版本，默认是9。值必须等于_onnx_main_opset或在_onnx_stable_opsets之内。具体可在torch/onnx/symbolic_helper.py中找到
           _retain_param_name=True, 
           do_constant_folding=True, 
           example_outputs=None, 
           strip_doc_string=True, 
           dynamic_axes=None,  # 设置动态的维度
           keep_initializers_as_inputs=None, 
           custom_opsets=None, 
           enable_onnx_checker=True, 
           use_external_data_format=False): 
```

### 代码示例
```
# pth模型转换到onnx
def export_to_onnx(dummy_input, weights, f=None, opset_version=12, simplify=True, dynamic=False):
    try:
        import onnx

        print(f'{weights} starting export with onnx {onnx.__version__}...')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = hrnet(base_channel=32, num_joint=1)
        ckpt = torch.load(weights, map_location='cpu')["state_dict"]
        model.load_state_dict(ckpt, strict=True)
        model = model.to(device)

        # dummy_input = torch.randn(3, 3, 448, 448, device=device)
        dummy_input = dummy_input.to(device)

        if f is None:
            f = weights.replace("pth", "onnx")

        # model.eval()
        torch.onnx.export(
            model,
            dummy_input,
            f,
            opset_version=opset_version,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,448,448)
                          'output': {0: 'batch', 2: 'heatmap_height', 3: 'heatmap_width'}  # shape(1,1,112,112)
                          } if dynamic else None)
        print("export succeed")

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # print(onnx.helper.printable_graph(model_onnx.graph))  # print
        # Simplify
        if simplify:
            try:
                import onnxsim

                print(f'{f} simplifying with onnx-simplifier {onnxsim.__version__}...')
                model_onnx, check = onnxsim.simplify(
                    model_onnx,
                    dynamic_input_shape=dynamic,
                    input_shapes={'images': list(dummy_input.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                print(f'{weights} simplifier failure: {e}')
        print(f'{weights} export success, saved as {f} ')
    except Exception as e:
        print(f'{weights} export failure: {e}')
```
