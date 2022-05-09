import glob
import json
import logging
import os.path
import time
import timeit
from pathlib import Path

import numpy as np
import argparse
import yaml
import shutil
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from models.network import build_model

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']


def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


@torch.no_grad()
def run(
        cfg='config/example.yaml',  # 配置文件，主要用于读取模型配置
        weights='best_model.pth',  # 模型路径
        source='./data/test',  # 测试数据路径，可以是文件夹，可以是单张图片
        use_cuda=True,  # 是否使用cuda
        view_img=False,  # 是否可视化测试图片
        save_mask=True,  # 是否将保存mask
        palette_path="palette.json",
        project='result'  # 结果输出路径
):
    if isinstance(cfg, dict):
        config = cfg
    else:
        import yaml
        yaml_file = Path(cfg).name
        with open(cfg) as f:
            config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    if save_mask:
        os.makedirs(project, exist_ok=True)

    # Load model
    assert os.path.exists(weights), "model path: {} does not exists".format(weights)
    num_classes = config["train"]["num_classes"] + 1
    model = build_model(config, num_classes, pretrain=False)
    checkpoint = torch.load(weights, map_location='cpu')["model"]
    model.load_state_dict(checkpoint, strict=True)
    model.eval().to(device)

    # 调色板，用于给mask上色
    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v

    # Run once
    y = model(torch.rand(1, 3, 480, 480).to(device))

    # Data transform
    data_transform = transforms.Compose([transforms.Resize(520),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                              std=(0.229, 0.224, 0.225))])

    # Load img
    assert os.path.exists(source), "data source: {} does not exists".format(source)
    if os.path.isdir(source):
        files = sorted(glob.glob(os.path.join(source, '*.*')))
    elif os.path.isfile(source):
        # img = Image.open(source)
        # if img.mode != 'RGB':
        #     img = img.convert('RGB')
        files = [source]
    else:
        raise Exception(f'ERROR: {source} does not exist')

    images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
    images = tqdm(images)
    for img_path in images:
        full_img = Image.open(img_path)
        img = data_transform(full_img)
        img = torch.unsqueeze(img, dim=0)
        t1 = time_sync()
        output = model(img.to(device))
        t2 = time_sync()

        print("inference time: {}".format(t2 - t1))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)

        file_name = img_path.split(os.sep)[-1][:-4]
        mask = Image.fromarray(prediction)
        mask.putpalette(pallette)
        mask.resize((full_img.size[1], full_img.size[0]), resample=Image.NEAREST)

        if view_img:
            fig, ax = plt.subplots(1, 2)
            ax[0].set_title('Input image')
            ax[0].imshow(full_img)
            ax[1].set_title(f'Output mask')
            ax[1].imshow(mask)
            plt.xticks([]), plt.yticks([])
            plt.show()

        if save_mask:
            mask.save(os.path.join(project, "{}.png".format(file_name)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--cfg', type=str, default='config/example.yaml',
                        help='experiment configure file name')
    parser.add_argument('--weights', '-w', default='best.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--source', '-i', default='data/test',
                        help='Filenames of input images/dir')
    parser.add_argument('--use_cuda', default=True, action='store_true', help='Use cuda to predict')
    parser.add_argument('--view_img', '-v', action='store_true', default=False,
                        help='Visualize the images as they are processed')
    parser.add_argument('--save-mask', '-sm', action='store_true', default=True, help='Whether save the output masks')
    parser.add_argument('--palette_path', default='palette.json', help='Plot Mask by different color')
    parser.add_argument('--project', '-p', metavar='OUTPUT', default='result', help='Output of img path')

    opt = parser.parse_args()
    run(**vars(opt))
