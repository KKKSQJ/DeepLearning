from copy import deepcopy
import yaml
import os
from tqdm import tqdm
import sys
from pathlib import Path
import argparse
import logging
import time
import warnings

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import FewShot, FSSDataset
from models import SSPNet
from utils import select_device, count_params, set_seed, mIOU, increment_path, Evaluator, Visualizer, to_cpu, to_cuda, \
    AverageMeter

warnings.filterwarnings("ignore")

# 获取脚本的绝对路径
FILE = Path(__file__).absolute()
# 将该脚本的父路径添加到系统路径中，方便脚本找到对应模快
sys.path.append(FILE.parents[0].as_posix())


def parser_args():
    parser = argparse.ArgumentParser(description="Few-shot Segmentation")
    parser.add_argument("--config-path", type=str, help='path of config file', default='config/test.yaml')
    # 其他参数去example.yaml中进行配置，下面参数是为了方便在服务器上进行跑
    parser.add_argument('--data-path', type=str, help='data path')
    parser.add_argument('--benchmark', type=str, default='fewshot')
    parser.add_argument("--device", type=str, help="device = 'cpu' or '0' or '0,1,2,3'")
    parser.add_argument('--project', default='runs/predict', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--weights', type=str, help='path of model weights', required=True)
    parser.add_argument('--fold', type=int, help='fold means classes,0 or 1 or 2 or 3')
    parser.add_argument('--shot', type=int, help='N way K shot 1, 2, 3, 4, 5')
    parser.add_argument('--test-snapshot', type=int, help='test snapshot', default=1000)
    parser.add_argument('--num-class', type=int, help='num classes')
    parser.add_argument('--vis', action='store_true', help='viusalize')
    parser.add_argument('--use-original-size', action='store_true')
    parser.add_argument('--input-size', default=[400, 400], help='test image size')
    # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser_args = parser.parse_args()

    with open(vars(parser_args)["config_path"], "r", encoding='utf-8') as config_file:
        config = yaml.full_load(config_file)

    if parser_args.data_path is not None:
        config["data"]["data_path"] = parser_args.data_path
    if parser_args.device is not None:
        config["test"]["device"] = parser_args.device
    if parser_args.project is not None:
        config["test"]["project"] = parser_args.project
    if parser_args.name is not None:
        config["test"]["name"] = parser_args.name
    if parser_args.weights is not None:
        config["test"]["weights"] = parser_args.weights
    if parser_args.fold is not None:
        config["test"]["fold"] = parser_args.fold
    if parser_args.shot is not None:
        config["test"]["shot"] = parser_args.shot
    if parser_args.test_snapshot is not None:
        config["test"]["test_snapshot"] = parser_args.test_snapshot
    if parser_args.num_class is not None:
        config["test"]["num_class"] = parser_args.num_class
    if parser_args.vis is not None:
        config["test"]["vis"] = parser_args.vis
    if parser_args.use_original_size is not None:
        config["test"]["use_original_size"] = parser_args.use_original_size
    if parser_args.input_size is not None:
        config["test"]["input_size"] = parser_args.input_size

    return config


def test(model, dataloader, config):
    r""" Test HSNet """

    # Freeze randomness during testing for reproducibility
    model.eval()
    set_seed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        # 1. Hypercorrelation Squeeze Networks forward pass
        batch = to_cuda(batch)
        img_s_list = batch['support_imgs']
        mask_s_list = batch['support_masks']
        img_q = batch['query_img']

        if not isinstance(img_s_list,list):
            img_s_list = [i.unsqueeze(0) for i in img_s_list.squeeze(0)]
            mask_s_list = [i.unsqueeze(0) for i in mask_s_list.squeeze(0)]

        # print(img_q.shape, mask_s_list.shape, img_s_list.shape)
        pred_mask = model(img_s_list, mask_s_list, img_q, None)[0]  # model.module.predict_mask_nshot(batch, nshot=nshot)
        pred_mask = torch.argmax(pred_mask, dim=1)

        assert pred_mask.size() == batch['query_mask'].size()

        # 2. Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=50)

        # Visualize predictions
        if Visualizer.visualize:
            Visualizer.visualize_prediction_batch(batch['support_imgs'], batch['support_masks'],
                                                  batch['query_img'], batch['query_mask'],
                                                  pred_mask, batch['class_id'], idx,
                                                  area_inter[1].float() / area_union[1].float())

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou


def main(config):
    print(config)

    # Create logging
    project = config["test"]["project"]
    name = config["test"]["name"]
    exist_ok = config["test"]["exist_ok"]
    save_dir = Path(str(increment_path(Path(project) / name, exist_ok=exist_ok)))
    save_dir.mkdir(parents=True, exist_ok=True)
    logging_path = str(save_dir / "test.log")
    logging.basicConfig(filename=logging_path, level=logging.INFO, filemode="w+")

    with open(save_dir / 'test.yaml', 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # device
    device = config["test"]["device"]
    bs = config["test"]["batch_size"]
    device = select_device(device, batch_size=bs)

    # model
    model_name = config["test"]["model_name"]
    refine = config["test"]["refine"]
    model = SSPNet(model_name, refine=refine)
    logging.info(model)
    logging.info('\nModel Params: %.1fM' % count_params(model))

    # load weights
    weights = config["test"]["weights"]
    if weights is not None and os.path.exists(weights):
        # 加载模型权重
        logging.info(f"Load pretrain weights from {weights}")
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model.load_state_dict(ckpt, strict=False)  # load

    model = torch.nn.DataParallel(model).to(device)
    # model = model.to(device)

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(config["test"]["vis"])

    # Dataset initialization
    FSSDataset.initialize(img_size=config["test"]["input_size"], datapath=config["data"]["data_path"],
                          use_original_imgsize=config["test"]["use_original_size"])
    dataloader_test = FSSDataset.build_dataloader(config["test"]["benchmark"], bs, config["test"]["num_workers"],
                                                  config["test"]["fold"], 'test', config["test"]["shot"],config["test"]["num_class"])

    # Test HSNet
    with torch.no_grad():
        test_miou, test_fb_iou = test(model, dataloader_test, config)
    logging.info(
        'Fold %d mIoU: %5.2f \t FB-IoU: %5.2f' % (config["test"]["fold"], test_miou.item(), test_fb_iou.item()))
    logging.info('==================== Finished Testing ====================')


if __name__ == '__main__':
    config = parser_args()
    main(config)

# import os
#
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
#
# path = 'Perseverance_124.jpg'
# img = cv2.imread(path,0)
# edges = cv2.Canny(img, 10, 100,9)
# #高斯平滑矩阵长与宽都为5，高斯矩阵尺寸越大，标准差越大，处理过的图像模糊程度越大。
# blur_ksize = 9
# #gray 进过灰度化处理的图像
# blur_gray = cv2.GaussianBlur(edges, (blur_ksize, blur_ksize), 0, 0)
#
# # 中值滤波，去除椒盐噪点
# # img_median = cv2.medianBlur(edges, 3)
# # cv2.imshow("img", img)
# # cv2.imshow("edges", edges)
# # cv2.waitkey()
# plt.subplot(131),plt.imshow(img,cmap = 'gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(132),plt.imshow(edges,cmap = 'gray')
# plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(133),plt.imshow(img_median,cmap = 'gray')
# plt.title('Blur Image'), plt.xticks([]), plt.yticks([])
# plt.show()
