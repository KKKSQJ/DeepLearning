import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

from torch.utils.tensorboard import SummaryWriter

from data_utils import preprocessing


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    """
    SSIM dissimilarity measure
    Args:
        img1: predicted image
        img2: target image
    """
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def mean_SSIM(x, y):
    """
    Mean error over SSIM reconstruction
    """
    return ssim(x, y, window_size=11, size_average=True)


def mean_l1(x, y, mask=None):
    """
    Mean reconstruction error
    Args:
        x: predicted image
        y: target image
        mask: compute only on this points
    """
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.float32)
    return torch.sum(mask * torch.abs(x - y)) / torch.sum(mask)


def mean_SSIM_L1(x, y):
    return 0.85 * mean_SSIM(x, y) + 0.15 * mean_l1(x, y)


SUPERVISED_LOSS = {
    'mean_l1': mean_l1,
    # 'sum_l1': sum_l1,
    # 'mean_l2': mean_l2,
    # 'sum_l2': sum_l2,
    # 'mean_SSIM': mean_SSIM,
    'mean_SSIM_l1': mean_SSIM_L1,
    # 'ZNCC': zncc,
    # 'cos_similarity': cos_similarity,
    # 'smoothness': smoothness,
    # 'mean_huber': mean_huber,
    # 'sum_huber': sum_huber
}
PIXELWISE_LOSSES = {
    # 'l1': l1,
    # 'l2': l2,
    'SSIM': SSIM,
    # 'huber': huber,
    # 'ssim_l1': ssim_l1
}

ALL_LOSSES = dict(SUPERVISED_LOSS)
ALL_LOSSES.update(PIXELWISE_LOSSES)


def get_reprojection_loss(reconstruction_loss, multiScale=False, logs=False, weights=None, reduced=True):
    """
    Build a lmbda op to be used to compute a loss function using reprojection between left and right frame
    Args:
    	reconstruction_loss: name of the loss function used to compare reprojected and real image
    	multiScale: if True compute multiple loss, one for each scale at which disparities are predicted
    	logs: if True enable tf summary
    	weights: array of weights to be multiplied for the losses at different resolution
    	reduced: if true return the sum of the loss across the different scales, false to return an array with the different losses
    """
    if reconstruction_loss not in ALL_LOSSES.keys():
        print('Unrecognized loss function, pick one among: {}'.format(ALL_LOSSES.keys()))
        raise Exception('Unknown loss function selected')
    base_loss_function = ALL_LOSSES[reconstruction_loss]
    if weights is None:
        weights = [1] * 10

    def compute_loss(disparities, inputs):
        left = inputs['left']
        right = inputs['right']
        # normalize image to be between 0 and 1
        # left = left / 256.0
        # right = right / 256.0
        accumulator = []
        if multiScale:
            disp_to_test = len(disparities)
        else:
            disp_to_test = 1
        for i in range(disp_to_test):
            # rescale prediction to full resolution
            current_disp = disparities[-(i + 1)]
            disparity_scale_factor = float(left.shape[3]) / float(current_disp.shape[3])
            resized_disp = preprocessing.resize_to_prediction(current_disp, left) * disparity_scale_factor
            reprojected_left = preprocessing.warp_image(right, resized_disp)
            partial_loss = base_loss_function(reprojected_left, left)
            if logs:
                tb_writer = SummaryWriter(log_dir="runs")
                tb_writer.add_scalar('Loss_resolution', partial_loss, i)
            accumulator.append(weights[i] * partial_loss)
        if reduced:
            return sum(accumulator)
        else:
            return accumulator

    return compute_loss

if __name__ == '__main__':
    a = [2,5]
    b = torch.sum(a)
    a = torch.randn(1, 3, 5, 5)
    d = a*2.0
    b = torch.sum(a)
    c = a.sum()
    print(1)
