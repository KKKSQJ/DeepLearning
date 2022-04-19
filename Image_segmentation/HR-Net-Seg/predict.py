import glob
import json
import logging
import os.path
import timeit
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

from dataLoader.dataloader import build_loaders
from models.network import build_model
from utils.utils import seed_everything
from utils.modelsummary import get_model_summary
from dataLoader.my_dataset import My_DataSet

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']


def parse_args():
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        type=str,
                        default='config/example.yaml')

    # parser.add_argument('opts',
    #                     help="Modify config options using the command-line",
    #                     default=None,
    #                     nargs=argparse.REMAINDER)

    parser.add_argument('--weights',
                        help='model checkpoint path',
                        type=str,
                        default='best.pth')

    parser.add_argument('--source',
                        help='test data',
                        type=str,
                        default='data')

    parser.add_argument('--test_val',
                        help='if test val',
                        default=False)

    args = parser.parse_args()
    with open(vars(args)["cfg"], "r") as config_file:
        hyperparams = yaml.full_load(config_file)

    return hyperparams, args


def run(hyperparams, args):
    # seed
    seed_everything()

    # create logging
    logging_name = hyperparams["train"]["logging_name"]
    logging_dir = "predict/{}".format(logging_name)
    shutil.rmtree(logging_dir, ignore_errors=True, )
    os.makedirs(logging_dir, exist_ok=True, )
    logging_path = os.path.join(logging_dir, "predict.log")
    logging.basicConfig(filename=logging_path, level=logging.INFO, filemode="w+")

    # create  device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Use device: {}".format(device))

    # create model
    in_channel = hyperparams["model"]["extra"]["in_channel"]
    num_classes = hyperparams["dataset"]["num_classes"] + 1
    # build model中内置了加载模型函数
    model = build_model(in_channel=in_channel, num_classes=num_classes, config=hyperparams).to(device)
    logging.info("Build  model successful !")
    logging.info("model in_channel: {} num_classes: {}".format(in_channel, num_classes))
    if os.path.exists(args.weights):
        pretrained_dict = torch.load(args.weights, map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict["state_dict"].items() if k in model_dict.keys()}
        for k, _ in pretrained_dict.items():
            logging.info(
                '=> loading {} pretrained model {}'.format(k, args.weights))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # run once
    x = torch.randn(1, in_channel, hyperparams["test"]["image_size"][0], hyperparams["test"]["image_size"][1]).to(
        device)
    # x = torch.randn(1, in_channel, 128, 128).to(device)
    y = model(x)

    logging.info(get_model_summary(model, x))

    # inference val
    if args.test_val and hyperparams["dataset"]["val"]["image_path"] is not None:
        # prepare val data(image and mask)
        loader = build_loaders(hyperparams, mode='train')
        val_dataset = loader["val_dataset"]
        val_loader = loader["val_dataloader"]
        # todo
        # 模型验证

    # inference test
    # prepare test data (only image)
    elif args.source is not None:
        # 如果命令行输入了测试数据，则推理命令行的数据，否则推理yaml中test下的数据
        # load img
        assert os.path.exists(args.source), "data source: {} does not exists".format(args.source)
        if os.path.isdir(args.source):
            files = sorted(glob.glob(os.path.join(args.source, '*.*')))
        elif os.path.isfile(args.source):
            files = [args.source]
        else:
            raise Exception(f'ERROR: {args.source} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]

        test_dataset = My_DataSet(
            img_path=images,
            label_path=None,
            num_sample=None,
            multi_scale=False,
            flip=False,
            brightness=False,
            mode="test",
            ignore_label=-1,
            base_size=(),
            crop_size=(),
            downsample_rate=3.0,  # 缩放倍率，变为原图的1/3大小
        )

        loader_args = dict(batch_size=1, num_workers=0, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, shuffle=False, **loader_args)
        run_test(model,
                 test_dataloader,
                 multi_scale_predict=False,
                 save_pred=True,
                 view_img=True,
                 save_dir=logging_dir)

    else:
        if os.path.exists(hyperparams["dataset"]["test"]["image_path"]):
            loader = build_loaders(hyperparams, mode='test')
            test_dataloader = loader["test_dataloader"]
            test_dataset = loader["test_dataset"]
            run_test(model,
                     test_dataloader,
                     multi_scale_predict=False,
                     save_pred=True,
                     view_img=True,
                     save_dir=logging_dir)


def run_test(model,
             test_loader,
             multi_scale_predict=False,
             save_pred=True,
             view_img=True,
             save_dir="./",
             palette_path="palette.json"):
    model.eval()

    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v

    with torch.no_grad():
        for _, batch in enumerate(tqdm(test_loader)):
            image, size, name = batch
            size = size[0]
            if multi_scale_predict:
                pass
            else:
                pred = model(image.cuda())

            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.interpolate(pred, (size[0], size[1]), mode='bilinear', align_corners=False)

            prediction = pred.argmax(1).squeeze(0)
            prediction = prediction.to("cpu").numpy().astype(np.uint8)
            mask = Image.fromarray(prediction)
            mask.putpalette(pallette)

            if view_img:
                # mask.show()
                plt.imshow(mask)
                plt.show()

            if save_pred:
                save_path = os.path.join(save_dir, "test_result")
                os.makedirs(save_path, exist_ok=True)
                mask.save(os.path.join(save_path, "{}.png".format(name[0])))


if __name__ == '__main__':
    # Usage
    """
    python predict.py --cfg config/example.yaml
    """
    hyperparams, args = parse_args()
    run(hyperparams, args)
