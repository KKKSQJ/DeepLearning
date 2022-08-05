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

from dataset import FewShot
from models import SSPNet
from utils import select_device, count_params, set_seed, mIOU, increment_path

warnings.filterwarnings("ignore")

# 获取脚本的绝对路径
FILE = Path(__file__).absolute()
# 将该脚本的父路径添加到系统路径中，方便脚本找到对应模快
sys.path.append(FILE.parents[0].as_posix())


def parser_args():
    parser = argparse.ArgumentParser(description="Few-shot Segmentation")
    parser.add_argument("--config-path", type=str, help='path of config file', default='config/test.yaml')
    # 其他参数去example.yaml中进行配置，下面参数是为了方便在服务器上进行跑
    parser.add_argument("--device", type=str, help="device = 'cpu' or '0' or '0,1,2,3'")
    parser.add_argument('--project', default='runs/test', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--weights', type=str, help='path of model weights', required=True)
    parser.add_argument('--fold', type=int, help='fold means classes,0 or 1 or 2 or 3', default=0)
    parser.add_argument('--shot', type=int, help='N way K shot 1, 2, 3, 4, 5', default=1)
    parser.add_argument('--test-snapshot', type=int, help='test snapshot', default=1000)
    parser.add_argument('--num-class', default=20, type=int, help='num classes')
    # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser_args = parser.parse_args()

    with open(vars(parser_args)["config_path"], "r", encoding='utf-8') as config_file:
        config = yaml.full_load(config_file)

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

    return config


def evaluate(model, dataloader, device, config):
    tbar = tqdm(dataloader)
    num_classes = config["test"]["num_class"] + 1
    metric = mIOU(num_classes)

    for i, (img_s_list, mask_s_list, img_q, mask_q, cls, _, id_q) in enumerate(tbar):
        img_q, mask_q = img_q.to(device), mask_q.to(device)
        for k in range(len(img_s_list)):
            img_s_list[k], mask_s_list[k] = img_s_list[k].to(device), mask_s_list[k].to(device)
        cls = cls[0].item()

        with torch.no_grad():
            out_ls = model(img_s_list, mask_s_list, img_q, mask_q)
            pred = torch.argmax(out_ls[0], dim=1)

        pred[pred == 1] = cls
        mask_q[mask_q == 1] = cls

        metric.add_batch(pred.cpu().numpy(), mask_q.cpu().numpy())

        tbar.set_description("Testing mIOU: %.2f" % (metric.evaluate() * 100.0))
    logging.info("Testing mIOU: %.2f" % (metric.evaluate() * 100.0))

    return metric.evaluate() * 100.0


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

    # Create Data
    img_path = config["data"]["img_path"]
    mask_path = config["data"]["mask_path"]
    # crop_size = config["train"]["input_size"]
    fold = config["test"]["fold"]
    shot = config["test"]["shot"]
    test_snapshot = config["test"]["test_snapshot"]
    num_class = config["test"]["num_class"]
    nw = config["test"]["num_workers"]

    testset = FewShot(img_path, mask_path, None, fold, shot, test_snapshot, num_class, mode="val")
    testloader = DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True, num_workers=nw, drop_last=False)

    # device
    bs = config["test"]["batch_size"]
    device = config["test"]["device"]
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

    logging.info('\nEvaluating on 5 seeds.....')
    model.eval()

    total_miou = 0.0
    for seed in range(5):
        logging.info('\nRun %i:' % (seed + 1))
        set_seed(seed)

        miou = evaluate(model, testloader, device, config)
        total_miou += miou

    logging.info('\n' + '*' * 32)
    logging.info('Averaged mIOU on 5 seeds: %.2f' % (total_miou / 5))
    logging.info('*' * 32 + '\n')


if __name__ == '__main__':
    config = parser_args()
    main(config)
