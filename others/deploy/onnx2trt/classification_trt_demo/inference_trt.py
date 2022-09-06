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


def softmax(x):
    x_exp = np.exp(x)
    # 如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    return s


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


def data_preprocessing(file_path, size=(224, 224), mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]):
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


def inference(
        engine,
        data_source,
        input_names='input',
        output_names='output',
        size=(224, 224),
        mean=[0.406, 0.456, 0.485],
        std=[0.225, 0.224, 0.229],
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

        # result = softmax(output)
        # score, index = np.max(result, axis=1), np.argmax(result, axis=1)
        # print(score[0], index[0])

        score = torch.max(torch.softmax(output, dim=1))
        c = torch.argmax(output, dim=1)

        score = score.item()
        c = c.item()
        print(f"score:{score}, class:{c}")

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
    parser.add_argument('--data_source', type=str, default='goldfish_class_1.jpg')  # 测试数据源，可以是单张图片，可以是文件夹
    parser.add_argument('--input_names', type=str, default='input')
    parser.add_argument('--output_names', type=str, default='output')
    parser.add_argument('--cal_fps', action='store_true', default=False)
    return parser.parse_args()


if __name__ == '__main__':
    # inference(
    #     engine='resnet50.engine',
    #     data_source='goldfish_class_1.jpg',
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
        size=(224,224),
        cal_fps=args.cal_fps
    )
