"""
此脚本用于测试yolov5中一些操作运算
单独测试，方便自己理解
"""

import os
import random

from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch
import math

def make_grid(nx=10, ny=10):
    """生成特征图网格坐标"""
    # 输出的shape:(1,1,ny,nx,2)
    # [[(0,0),(0,1),(0,2)],
    #  [(1,0),(1,1),(1,2)]]
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

def dim_change():
    """测试tensor维度变化"""
    t = torch.tensor(np.random.randn(1, 255, 20, 20))
    t1 = t.view(1, 3, 85, 20, 20)
    t2 = t1.permute(0, 1, 3, 4, 2)
    t3 = t2.contiguous()
    return t3

def test_sigmoid():
    """测试sigmoid()"""
    # y = 1/1+exp(-x)
    t = torch.tensor(np.random.randn(1,2))
    t1 = t.sigmoid()
    t2 = 1/(1+math.exp(-t[...,0].numpy())+1e-15)
    t3 = 1/(1+math.exp(-t[...,1].numpy())+1e-15)
    return t1

# def make_point():
#     """预测框坐标反算"""
#     x = torch.tensor(np.random.randn(1, 255, 20, 20))
#     x = x.view(1, 3, 85, 20, 20).permute(0, 1, 3, 4, 2).contiguous()
#     y = x.sigmoid()
#     grid = make_grid()
#     stride = torch.tensor([32])
#     anchor_grid = torch.tensor([1])
#     y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * stride # xy
#     y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid  # wh


def test_mosaic():
    mosaic_boarder = [-320,-320]
    s = 640
    yc,xc = [int(random.uniform(-x,2*s+x)) for x in mosaic_boarder]
    img = np.full((s*2, s*2, 3), 114, dtype=np.uint8)
    print(1)

if __name__ == '__main__':
    a = make_grid()
    print(a)
    b = dim_change()
    print(b)
    c = test_sigmoid()
    print(c)
    # d = make_point()
    e = test_mosaic()