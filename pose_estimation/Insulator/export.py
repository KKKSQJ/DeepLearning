import numpy as np
import os

import torch
import onnxruntime
import cv2
from dataset import kp_transforms as transforms
from models import HighResolution as hrnet
from PIL import ImageDraw, ImageFont
from PIL.Image import Image
import PIL


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


# 模型可视化：Netron
"""
每个算子记录了算子属性、图结构、权重三类信息。

算子属性信息即图中 attributes 里的信息，对于卷积来说，算子属性包括了卷积核大小(kernel_shape)、卷积步长(strides)等内容。这些算子属性最终会用来生成一个具体的算子。
图结构信息指算子节点在计算图中的名称、邻边的信息。对于图中的卷积来说，该算子节点叫做 Conv_2，输入数据叫做 11，输出数据叫做 12。根据每个算子节点的图结构信息，就能完整地复原出网络的计算图。
权重信息指的是网络经过训练后，算子存储的权重信息。对于卷积来说，权重信息包括卷积核的权重值和卷积后的偏差值。点击图中 conv1.weight, conv1.bias 后面的加号即可看到权重信息的具体内容。

"""


# onnx 模型推理
def inference(img_path, onnx_weights, output, size=(448, 448)):
    assert os.path.exists(img_path)
    # hwc ->  chw
    img = cv2.imread(img_path)  # .astype(np.float32)
    height, width, c = img.shape
    # rgb->bgr
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # data transform
    data_transform = transforms.Compose(
        [transforms.AffineTransform(fixed_size=size),
         # transforms.KeypointToHeatMap(heatmap_hw=(size[0] // 4, size[1] // 4), keypoints_nums=1),
         # transforms.ToTensor(),
         # transforms.Normalize([0.616, 0.231, 0.393], [0.312, 0.288, 0.250])
         ])
    img, target = data_transform(img, {"box": [0, 0, width - 1, height - 1]})
    # cv2.imshow("a",img)
    # cv2.waitKey(0)
    # /255.0
    img = img / 255.
    # normalize
    mean = [0.616, 0.231, 0.393]
    std = [0.312, 0.288, 0.250]
    img = (img - mean) / std
    # float32
    img = img.astype(np.float32)

    # hwc->chw->bchw
    img = img.transpose(2, 0, 1)
    img = img[None, :]

    assert onnx_weights.endswith("onnx")
    ort_session = onnxruntime.InferenceSession(onnx_weights)
    ort_imgs = {'images': img}
    # b c h w
    ort_output = ort_session.run(["output"], ort_imgs)[0]

    # post
    batch_size, num_joints, h, w = ort_output.shape
    reshape_pred = ort_output.reshape(batch_size, num_joints, -1)
    points = np.where(reshape_pred > 0.6)[-1]

    # [30 31 32 33 34 34 35 35 36 36 37 37 38 38 38 38 39 39 40 40 41 41 41 42, 42 42 42 43 43 44 44 45 46 46 47 48]
    # [86 82 78 75 67 71 64 86 60 82 56 79 49 53 72 75 45 68 42 64 35 38 61 27, 31 54 58 23 51 44 47 40 33 37 30 26]
    final_preds = np.zeros((batch_size, num_joints, len(points), 3))

    points_x = points % w
    points_y = np.floor(points / w)

    final_preds[:, :, :, 2] = ort_output[:, :, points_y.astype(np.int32), points_x.astype(np.int32)]

    points_y = points_y * size[0] / (h - 1)
    points_x = points_x * size[1] / (w - 1)
    final_preds[:, :, :, 0] = points_x
    final_preds[:, :, :, 1] = points_y

    # draw point
    # bchw->chw->hwc
    img = img.squeeze().transpose(1, 2, 0)
    # *std+mean * 255
    img = (img * std + mean) * 255.
    # bgr->rgb
    img = img[:, :, ::-1]
    img = img.astype(np.uint8)

    for i, (point_x, point_y, score) in enumerate(final_preds[0][0]):
        cv2.circle(img, (int(point_x), int(point_y)), 2, (0, 0, 255), 2)

    # if output:
    #     cv2.save("")
    cv2.imshow("a", img)
    cv2.waitKey(0)
    return img



if __name__ == '__main__':
    # export_to_onnx(dummy_input=torch.randn(3, 3, 448, 448),
    #                weights="hrnet_best.pth",
    #                opset_version=12,
    #                simplify=True,
    #                dynamic=True)

    # onnx模型推理
    inference(img_path='000000.jpg', onnx_weights='hrnet_best.onnx', output='o')
