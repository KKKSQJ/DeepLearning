import sys
import os
import argparse
from pprint import pprint
import numpy as np
from loguru import logger

import torch
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

from trainers.trainer import Trainer


@logger.catch
def run(opt):
    trainer = Trainer(opt)
    trainer.train()


def parser_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./data', help='data path')
    parser.add_argument('--datatype', type=str, default='button')
    parser.add_argument('--mode', type=str, default='retrieval', help='retrieval or classification')
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--max-epoch', type=int, default=400)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--num-instances', type=int, default=4, help='train per class num')
    parser.add_argument('--weights', type=str, default='./model.pth', help='pretrain model path')
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--evaluate', type=bool, default=False)
    parser.add_argument('--save-fig', type=bool, default=False)
    parser.add_argument('--re-ranking', type=bool, default=False)
    parser.add_argument('--last-stride', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=30)
    parser.add_argument('--eval-step', type=int, default=10)
    parser.add_argument('--use-gpu', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--model-name', type=str, default='bfe')
    parser.add_argument('--h-ratio', type=float, default=0.3)
    parser.add_argument('--w-ratio', type=float, default=0.3)
    parser.add_argument('--global-feature-dim', type=int, default=512)
    parser.add_argument('--part-feature-dim', type=int, default=1024)
    parser.add_argument('--CrossEntropy-with-label-smooth', type=bool, default=True)
    parser.add_argument('--loss', type=str, default='triplet')
    parser.add_argument('--margin', type=float, default=2)
    parser.add_argument('--random-crop', type=bool, default=False)
    parser.add_argument('--adjust-lr', type=str, default='steplr2',help=' "cosine" or "steplr" or "steplr2"')

    args = parser.parse_args()

    return args


def state_dict(opt):
    return {k: getattr(opt, k) for k, _ in opt.__dict__.items()}


if __name__ == '__main__':
    args = parser_arg()
    run(opt=args)
