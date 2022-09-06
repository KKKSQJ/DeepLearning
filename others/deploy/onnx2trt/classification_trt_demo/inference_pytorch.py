import time
import torch
from torchvision.models import resnet50

import numpy as np
import os
import glob
from tqdm import tqdm
import cv2

IMG_FORMATS = ["jpg", "png", "jpeg"]


def data_preprocessing(file_path, size=(224, 224), mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]):
    # H W C BGR
    img = cv2.imread(file_path)
    # RESIZE
    img = cv2.resize(img, dsize=tuple(size))
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


@torch.no_grad()
def inference(
        weight,
        data_source,
        size=(224, 224),
        mean=[0.406, 0.456, 0.485],
        std=[0.225, 0.224, 0.229],
        cal_fps=False
):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    model = resnet50(pretrained=False)
    ckpt = torch.load(weight, map_location='cpu')
    model.load_state_dict(ckpt, strict=True)
    model = model.to(device)
    model.eval()

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

        t1 = time.perf_counter()
        output = model(torch.from_numpy(image).to(device))
        t2 = time.perf_counter()
        elapsed = t2 - t1

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
    parser.add_argument('--weight', type=str, default='model.pth')
    parser.add_argument('--data_source', type=str, default='goldfish_class_1.jpg')  # 测试数据源，可以是单张图片，可以是文件夹
    parser.add_argument('--cal_fps', action='store_true', default=False)
    return parser.parse_args()


if __name__ == '__main__':
    # inference(
    #     weight='resnet50-0676ba61.pth',
    #     data_source='goldfish_class_1.jpg',
    #     size=(224, 224),
    #     mean=[0.406, 0.456, 0.485],
    #     std=[0.225, 0.224, 0.229],
    #     cal_fps=True
    # )

    args = parser_args()
    inference(
        weight=args.weight,
        data_source=args.data_source,
        size=(224,224),
        mean=[0.406, 0.456, 0.485],
        std=[0.225, 0.224, 0.229],
        cal_fps=args.cal_fps
    )
