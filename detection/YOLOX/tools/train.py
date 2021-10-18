#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.core import Trainer, launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, configure_omp, get_num_devices


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=None, type=int, help="device for training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="plz input your experiment description file",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="resume training"
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=None,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        dest="cache",
        default=False,
        action="store_true",
        help="Caching imgs to RAM for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = Trainer(exp, args)
    trainer.train()


if __name__ == "__main__":
    """
    训练的时候需要注意的是：
        1.如果使用voc格式的数据进行训练，则使用-f exps/example/yolox_voc/yolox_voc_s.py
        需要对yolox_voc_s.py做以下修改：
        1.
            模型修改，默认yolox_s.pth
            对应参数 self.depth = 0.33   
                    self.width = 0.50
            如使用其他模型，则修改为对应的 depth,width
        2.
            类别修改。 对应参数 self.num_classes = 4（类别数量，不含背景类） 
            再修改yolox/data/voc_classes.py中的类别
        3. 
            数据存放路径修改
            对应参数self.data_dir = 'xxxx/name_dir' 
            需要绝对路径，且保持下面文件夹命名方式
        数据存放格式： 
                ---name_dir
                    -----train2007
                        ---------JPEGImages
                        ---------Annotations
                    -----test2007
                         ---------JPEGImages
                         ---------Annotations
        4.
            是否使用标准的VOC格式（保持self.standard_voc=False即可）
                对应参数self.standard_voc=False。详细参考标准的voc数据存放的文件命名
        5.
            修改图片的后缀
                如xx.png。则需要修改yolox/data/datasets/voc.py中  
                    class VOCDetection(Dataset) 中的
                        self._imgpath = os.path.join("%s", "JPEGImages", "%s.png")  # "%s.jpg"
                        
        6.
            训练命令：
            从头开始： python tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 1 -b 4 --fp16 -c yolox_s.pth
            接着训练： python tools/train.py -f exps/example/yolox_voc/yolox_voc_s.py -d 1 -b 4 -c YOLOX_outputs/yolox_voc_s/latest_ckpt.pth.tar -resume -start_epoch=100
    """

    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )
