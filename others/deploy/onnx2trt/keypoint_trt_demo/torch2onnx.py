"""
Pytorch 模型 转 ONNX模型

Model: HRnet
Task: 关键点检测
模型输入： 1 3 448 448
模型输出： 1 1 112 112 只有一个关键点

"""
import torch
from hrnet import HighResolution as hrnet


def export_to_onnx(weights, dummy_input=None, f=None, opset_version=12, simplify=True, dynamic=False):
    try:
        import onnx

        print(f'{weights} starting export with onnx {onnx.__version__}...')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = hrnet(base_channel=32, num_joint=1)
        ckpt = torch.load(weights, map_location='cpu')["state_dict"]
        model.load_state_dict(ckpt, strict=True)
        model = model.to(device)

        if dummy_input is None:
            dummy_input = torch.randn(3, 3, 448, 448)
        dummy_input = dummy_input.to(device)

        if f is None:
            f = weights.replace("pth", "onnx")

        # model.eval()
        torch.onnx.export(
            model,
            dummy_input,
            f,
            opset_version=opset_version,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,448,448)
                          'output': {0: 'batch'}  # shape(1,1,112,112)
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
                    input_shapes={'input': list(dummy_input.shape)} if dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                print(f'{weights} simplifier failure: {e}')
        print(f'{weights} export success, saved as {f} ')
    except Exception as e:
        print(f'{weights} export failure: {e}')


def parser_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='model.pth')
    parser.add_argument('--f', type=str, default='model.onnx')
    parser.add_argument('--opset_version', type=int, default=12)
    parser.add_argument('--simplify', action='store_true', default=False)
    parser.add_argument('--dynamic', action='store_true', default=False)
    return parser.parse_args()


if __name__ == '__main__':
    # weights = 'hrnet.pth'
    #
    # export_to_onnx(weights=weights,
    #                dummy_input=torch.randn(1, 3, 224, 224),
    #                f='model.onnx',
    #                opset_version=12,
    #                simplify=True,
    #                dynamic=True)

    args = parser_args()
    export_to_onnx(
        weights=args.weight,
        f=args.f,
        opset_version=args.opset_version,
        simplify=args.simplify,
        dynamic=args.dynamic
    )
