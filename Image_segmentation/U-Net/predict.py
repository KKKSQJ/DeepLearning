import argparse
import glob
import json
import logging
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from dataLoader import BasicDataset
from models import UNet
from utils.utils import plot_img_and_mask

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']


def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


@torch.no_grad()
def run(
        weights='best_model.pth',  # 模型路径
        source='./data/test',  # 测试数据路径，可以是文件夹，可以是单张图片
        use_cuda=True,  # 是否使用cuda
        view_img=False,  # 是否可视化测试图片
        save_mask=True,  # 是否将保存mask
        mask_threshold=0.5,
        bilinear=False,
        scale=0.5,
        palette_path="./utils/palette.json",
        project='result'  # 结果输出路径

):
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    if save_mask:
        os.makedirs(project, exist_ok=True)

    # load model
    assert os.path.exists(weights), "model path: {} does not exists".format(weights)
    model = UNet(in_channel=3, classes=2, bilinear=bilinear)
    model.load_state_dict(torch.load(weights, map_location=device), strict=True)
    model.eval().to(device)

    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v

    # run once
    y = model(torch.rand(1, 3, 224, 224).to(device))

    # load img
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
        # # cv2:BGR
        # img = cv2.imread(img_path)
        # # Convert
        # # BGR转为RGB格式，channel轴换到前面
        # img0 = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        # # 将数组内存转为连续，提高运行速度，(不转的话也可能会报错)
        # img0 = np.ascontiguousarray(img0)
        # img0 = img0.astype('float32')
        # img0 /= 255.
        # img_tensor = torch.from_numpy(img0)
        # if len(img_tensor.shape) == 3:
        #     img_tensor = img_tensor[None]  # 等价于img_tensor = img_tensor.unsqueeze(0)
        """
        # toTensor
        # 第71行到78行等价于下面两句话
        img = cv2.imread(img_path)
        img = transforms.ToTensor()(img)
        """
        full_img = Image.open(img_path)
        img = torch.from_numpy(BasicDataset.preprocess(full_img, scale, is_mask=False))
        img_tensor = img[None].to(device, dtype=torch.float32)
        t1 = time_sync()
        pred = model(img_tensor.to(device))
        t2 = time_sync()
        if model.classes > 1:
            probs = F.softmax(pred, dim=1)[0]
        else:
            probs = torch.sigmoid(pred)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()
        if model.classes == 1:
            mask = (full_mask > mask_threshold).numpy()
        else:
            # mask = Image.fromarray(full_mask.argmax(dim=0).numpy().astype(np.uint8))
            mask = F.one_hot(full_mask.argmax(dim=0), model.classes).permute(2, 0, 1).numpy()

        print("inference time: {:.5f}s Done.".format((t2 - t1)))

        # 可视化图片
        if view_img:
            plot_img_and_mask(full_img, mask)

        if save_mask:
            file_name = img_path.split(os.sep)[-1][:-4]
            mask = mask_to_image(mask)
            # mask.putpalette(pallette)
            mask.save(os.path.join(project, "{}.png".format(file_name)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--weights', '-w', default='checkpoint_epoch5.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--source', '-i', default=r'E:\dataset\archive\train_images',
                        help='Filenames of input images/dir',
                        required=True)
    parser.add_argument('--use_cuda', default=True, action='store_true', help='Use cuda to predict')
    parser.add_argument('--view_img', '-v', action='store_true', default=False,
                        help='Visualize the images as they are processed')
    parser.add_argument('--save-mask', '-sm', action='store_true', default=True, help='Whether save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--palette_path', default='./utils/palette.json', help='Plot Mask by different color')
    parser.add_argument('--project', '-p', metavar='OUTPUT', default='result', help='Output of img path')

    opt = parser.parse_args()
    run(**vars(opt))
