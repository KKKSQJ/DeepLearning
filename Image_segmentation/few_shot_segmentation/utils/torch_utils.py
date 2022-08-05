import torch
import os
import logging
import platform
import time
import random
import numpy as np
import re
from pathlib import Path
import glob

LOGGER = logging.getLogger(__name__)


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'torch {torch.__version__} '  # string
    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    LOGGER.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    return torch.device('cuda:0' if cuda else 'cpu')


def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


def set_seed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mean(x):
    return sum(x) / len(x) if len(x) > 0 else 0.0


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch


def to_cpu(tensor):
    if isinstance(tensor,list):
        tensor = torch.stack(tensor)
    return tensor.detach().clone().cpu()


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    """根据文件夹中已有的文件名，自动获得新路径或文件名"""
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        # 获取文件后缀
        suffix = path.suffix
        # 去掉后缀的path
        path = path.with_suffix('')
        # 获取所有以{path}{sep}开头的文件
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # 在dirs中找到以数字结尾的文件
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        # 获取dirs文件结尾的数字
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        # 最大的数字+1
        n = max(i) + 1 if i else 2  # increment number
        # 设置新文件的文件名
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    # 获取文件路径并创建
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path


if __name__ == '__main__':
    device = select_device("0", batch_size=16)
    print(device)
