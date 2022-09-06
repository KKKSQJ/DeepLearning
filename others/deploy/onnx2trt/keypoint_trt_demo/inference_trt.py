import time
from typing import Union, Optional, Sequence, Dict, Any

import torch
import tensorrt as trt

import numpy as np
import os
import glob
from tqdm import tqdm
import cv2

IMG_FORMATS = ["jpg", "png", "jpeg"]


class TRTWrapper(torch.nn.Module):
    def __init__(self, engine: Union[str, trt.ICudaEngine],
                 output_names: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self.engine = engine
        if isinstance(self.engine, str):
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                with open(self.engine, mode='rb') as f:
                    engine_bytes = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()
        names = [_ for _ in self.engine]
        input_names = list(filter(self.engine.binding_is_input, names))
        self._input_names = input_names
        self._output_names = output_names

        if self._output_names is None:
            output_names = list(set(names) - set(input_names))
            self._output_names = output_names

    def forward(self, inputs: Dict[str, torch.Tensor]):
        assert self._input_names is not None
        assert self._output_names is not None
        bindings = [None] * (len(self._input_names) + len(self._output_names))
        profile_id = 0
        for input_name, input_tensor in inputs.items():
            # check if input shape is valid
            profile = self.engine.get_profile_shape(profile_id, input_name)
            assert input_tensor.dim() == len(
                profile[0]), 'Input dim is different from engine profile.'
            for s_min, s_input, s_max in zip(profile[0], input_tensor.shape,
                                             profile[2]):
                assert s_min <= s_input <= s_max, \
                    'Input shape should be between ' \
                    + f'{profile[0]} and {profile[2]}' \
                    + f' but get {tuple(input_tensor.shape)}.'
            idx = self.engine.get_binding_index(input_name)

            # All input tensors must be gpu variables
            assert 'cuda' in input_tensor.device.type
            input_tensor = input_tensor.contiguous()
            if input_tensor.dtype == torch.long:
                input_tensor = input_tensor.int()
            self.context.set_binding_shape(idx, tuple(input_tensor.shape))
            bindings[idx] = input_tensor.contiguous().data_ptr()

            # create output tensors
        outputs = {}
        for output_name in self._output_names:
            idx = self.engine.get_binding_index(output_name)
            dtype = torch.float32
            shape = tuple(self.context.get_binding_shape(idx))

            device = torch.device('cuda')
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            bindings[idx] = output.data_ptr()
        self.context.execute_async_v2(bindings,
                                      torch.cuda.current_stream().cuda_stream)
        return outputs


def data_preprocessing(file_path, size=(448, 448), mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]):
    # H W C BGR
    img = cv2.imread(file_path)
    # RESIZE
    img = cv2.resize(img, dsize=size)
    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalization
    mean = np.array(mean).astype(np.float32)
    std = np.array(std).astype(np.float32)
    img = (img / 255. - mean) / std
    # B H W C
    img = np.expand_dims(img, axis=0)
    # B H W C -> B C H W
    img = np.transpose(img, (0, 3, 1, 2)).astype(np.float32)
    return img

# 得到最终的预测结果，输出1 1 n 3 的tensor,n表示有n个点、3：（1,2）表示（x,y），3表示置信度
def get_keypoints_preds(preds, img_size, thresh=0.6, max_kp=50):
    assert len(img_size) == 2
    # preds = torch.sigmoid(preds)
    # # 关键点非极大值抑制
    # preds = _nms(preds)

    batch_size, num_joints, h, w = preds.shape
    output = []

    reshaped_preds = preds.reshape(batch_size, num_joints, -1)
    for b in range(batch_size):
        for c in range(num_joints):
            reshaped_pred = reshaped_preds[b][c]
            p = torch.where(reshaped_pred > thresh)[0]
            p = p[reshaped_pred[p].argsort(descending=True)]
            confidence = reshaped_pred[p]
            if not p.shape[0]:
                continue
            elif p.shape[0] > max_kp:
                # p = p[reshaped_pred[p].argsort(descending=True)][:max_kp]
                p = p[:max_kp]
            p_x = p % w
            p_y = torch.floor(p / w)
            t = torch.zeros(len(p), 4).to(preds)
            # 在网络输入图像上的像素x
            t[..., 0] = p_x.long() * img_size[1] / (w - 1)
            # 在网络输入图像上的像素y
            t[..., 1] = p_y.long() * img_size[0] / (h - 1)
            # 置信度
            t[..., 2] = reshaped_pred[p]
            # 关键点类别
            t[..., 3] = c

        if p.shape[0]:
            output.append(t)

    # output:[bs] 表示bs张图片的关键点信息
    # bs:[n,4] 表示这张图上有n个点，每个点有4个信息，分别是【 x y 置信度 关键点的类别】
    return output, thresh

def draw_pic(image, points):
    for i, p in enumerate(points):
        pos_x = int(p.cpu().numpy()[0])
        pos_y = int(p.cpu().numpy()[1])
        conf = float(p.cpu().numpy()[2])
        cv2.circle(image, (pos_x, pos_y), 1, (0, 255, 0), 2)


#  b c h w
def decode_image(image, mean, std):
    img = image.squeeze().transpose(1, 2, 0)
    # *std+mean * 255
    img = (img * std + mean) * 255.
    # bgr->rgb
    img = img[:, :, ::-1]
    img = img.astype(np.uint8)
    return img

def inference(
        engine,
        data_source,
        input_names='input',
        output_names='output',
        size=(224, 224),
        mean=[0.406, 0.456, 0.485],
        std=[0.225, 0.224, 0.229],
        show_img=False,
        cal_fps=True
):
    # Device
    device = torch.device('cuda')
    # Load engine
    assert os.path.exists(engine)
    model = TRTWrapper(engine, [output_names])
    # Load img
    assert os.path.exists(data_source), "data source: {} does not exists".format(data_source)
    if os.path.isdir(data_source):
        files = sorted(glob.glob(os.path.join(data_source, '*.*')))
    elif os.path.isfile(data_source):
        files = [data_source]
    else:
        raise Exception(f'ERROR: {data_source} does not exist')

    images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
    # images = tqdm(images)

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    fps = 0
    pure_inf_time = 0
    if cal_fps:
        if len(images) < num_warmup:
            images *= 100

    for index, image_file in (enumerate(images)):
        # torch.cuda.synchronize()
        image = data_preprocessing(image_file, size=size, mean=mean, std=std)
        input_shape = {input_names: torch.from_numpy(image).to(device)}
        t1 = time.perf_counter()
        output = model(input_shape)[output_names]
        t2 = time.perf_counter()
        elapsed = t2 - t1
        out, _ = get_keypoints_preds(output, img_size=size, thresh=0.6, max_kp=50)

        # 带后处理，fps为65.5。每张图片15.3ms .
        # 不带后处理 fps365.0。每张图片2.7ms

        draw_img = decode_image(image, mean, std)
        if show_img and len(out[0]) > 0:
            draw_pic(draw_img, out[0])
            cv2.imshow('img', draw_img)
            cv2.waitKey()


        # fps
        if index >= num_warmup:
            pure_inf_time += elapsed

            if (index + 1) % 100 == 0:
                fps = (index + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{index + 1:<3}/ {len(images)}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (index + 1) == len(images) and (index + 1) > num_warmup:
            fps = (index + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                flush=True)

        print("time:{}".format(t2 - t1))
        # print(output)


def parser_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='model.engine')
    parser.add_argument('--data_source', type=str, default='000000.jpg')  # 测试数据源，可以是单张图片，可以是文件夹
    parser.add_argument('--input_names', type=str, default='input')
    parser.add_argument('--output_names', type=str, default='output')
    parser.add_argument('--cal_fps', action='store_true', default=False)
    return parser.parse_args()


if __name__ == '__main__':
    # inference(
    #     engine='model.engine',
    #     data_source='000000.jpg',
    #     input_names='input',
    #     output_names='output',
    #     size=(224, 224),
    #     cal_fps=True)

    args = parser_args()
    inference(
        engine=args.engine,
        data_source=args.data_source,
        input_names=args.input_names,
        output_names=args.output_names,
        size=(448, 448),
        cal_fps=args.cal_fps
    )
