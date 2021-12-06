import sys
import os
import argparse
from loguru import logger

import torch

from trainers.trainer import Trainer


def parser_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='./dataset', help='data path')
    parser.add_argument('--data-type', type=str, default='voc', help='coco or voc or ...')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='size of each image batch')
    parser.add_argument('--warmup_steps', type=int, default=501, help='warm up strps')
    parser.add_argument('--workers', type=int, default=4, help='number of cpu threads to use during batch generation')
    parser.add_argument('--gpu-ids', type=str, default='0', help='id of gpu to use during training "0" or "0,1,2,3" ')
    parser.add_argument('--lr-init', type=float, default=2e-3, help='Initial learning rate')
    parser.add_argument('--lr-end', type=float, default=2e-5, help='end learning rate')
    parser.add_argument('--optimizer', type=str, default='SGD', help='SGD or Adam')
    parser.add_argument('--pretrain-weights', type=str, default=None, help='the path of pretrain weights')
    parser.add_argument('--save-path', type=str, default='./checkpoint', help='the path of output weights')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use-gpu', type=bool, default=True, help='if use gpu')
    parser.add_argument('--per-eval',type=int,default=1,help='eval ')
    args = parser.parse_args()

    return args


def state_dict(opt):
    return {k: getattr(opt, k) for k, _ in opt.__dict__.items()}


@logger.catch
def run(opt):
    if opt.use_gpu and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids

    trainer = Trainer(opt)
    trainer.train()


if __name__ == '__main__':
    args = parser_arg()
    run(opt=args)
