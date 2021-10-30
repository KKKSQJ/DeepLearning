import json

import torch
from torchvision import transforms

import cv2
import sys
import argparse
import os
from PIL import Image
import numpy as np
import glob
import time
import shutil
from tqdm import tqdm

from models.network import efficientnet_b0

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']


def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


@torch.no_grad()
def run(
        model_name='efficientb0',  # 网络名字
        weights='best_model.pth',  # 模型路径
        source='./data/test',  # 测试数据路径，可以是文件夹，可以是单张图片
        use_cuda=True,  # 是否使用cuda
        view_img=False,  # 是否可视化测试图片
        save_txt=True,  # 是否将结果保存到txt
        project='runs/result'  # 结果输出路径

):
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    if save_txt:
        if os.path.exists(project):
            shutil.rmtree(project)
        os.makedirs(project)
        f = open(project + "/result.txt", 'w')

    img_size = {"B0": 224,
                "B1": 240,
                "B2": 260,
                "B3": 300,
                "B4": 380,
                "B5": 456,
                "B6": 528,
                "B7": 600}
    num_model = "B0"

    data_transform = transforms.Compose([transforms.RandomResizedCrop(img_size[num_model]),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # load model
    assert os.path.exists(weights), "model path: {} does not exists".format(weights)
    model = efficientnet_b0(num_classes=5)

    model.load_state_dict(torch.load(weights, map_location=device), strict=True)
    model.eval().to(device)

    # run once
    y = model(torch.rand(1, 3, img_size[num_model], img_size[num_model]).to(device))

    # load img
    assert os.path.exists(source), "data source: {} does not exists".format(source)
    if os.path.isdir(source):
        files = sorted(glob.glob(os.path.join(source, '*.*')))
    elif os.path.isfile(source):
        files = [source]
    else:
        raise Exception(f'ERROR: {source} does not exist')

    images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
    # images = tqdm(images)
    for img_path in images:
        img = Image.open(img_path)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        t1 = time_sync()
        pred = model(img.to(device))
        t2 = time_sync()
        pred_class = torch.max(pred, dim=1)[1]
        c = pred_class.cpu().numpy().item()
        prob = torch.squeeze(torch.softmax(pred, dim=1)).cpu().numpy()[int(c)]
        print("class: {}\tprob: {:.3}\tinference time: {:.5f}s Done.".format(class_indict[str(c)],prob,(t2 - t1)))


        # 可视化图片
        if view_img:
            img = cv2.imread(img_path)
            cv2.imshow("image", img)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, class_indict[str(c)], (14, 14), font, 1, (0, 0, 255), 3)
            cv2.waitKey()
        if save_txt:
            file_name = img_path.split(os.sep)[-1]
            f.write("{} {}\n".format(file_name, class_indict[str(c)]))
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='efficientb0')
    parser.add_argument('--weights', type=str, default='best_model.pth', help='the model path')
    parser.add_argument('--source', type=str, default='./data/test', help='test data path')  # /0_00001.jpg
    parser.add_argument('--use-cuda', type=bool, default=True)
    parser.add_argument('--view-img', type=bool, default=False)
    parser.add_argument('-s', '--save-txt', type=bool, default=True)
    parser.add_argument('--project', type=str, default='runs/result', help='output path')
    opt = parser.parse_args()
    run(**vars(opt))
