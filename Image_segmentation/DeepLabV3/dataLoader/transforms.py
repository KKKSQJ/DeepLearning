import numbers
from pathlib import Path

import numpy as np
import random

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image

"""
这个博客详细记载了torchvision.transforms.functional常用的数据增强
https://www.cnblogs.com/ghgxj/p/14219097.html

温馨提示：
    1.在模型训练过程中，每个epoch做数据增强，可以增加样本数量。
    因为，每一个epoch虽然训练的样本数量不变，但是每一个epoch因为transform不同而产生不一样的训练样本，相当于增加了样本数量。
    2.每一个batch的样本使用的是相同的数据增强方式。
    因为loader会将N个样本打包成一个batch，送进模型。因此，这一批数据的数据增强方式是一样。
    不同批次的数据因为transfrom不同，其变化不同。
"""


def pad_if_smaller(img, size, fill=0):
    # 如果图像最小边长小于给定size，则用数值fill进行padding
    min_size = min(img.size)
    ow, oh = img.size

    if isinstance(size, int):
        if min_size < size:
            padh = size - oh if oh < size else 0
            padw = size - ow if ow < size else 0
            img = F.pad(img, padding=[0, 0, int(padw), int(padh)], fill=fill)

    elif len(size) == 2:
        h, w = size
        padh = h - oh if oh < h else 0
        padw = w - ow if ow < w else 0
        img = F.pad(img, padding=[0, 0, int(padw), int(padh)], fill=fill)
    return img


# 随机缩放
# class RandomResize(object):
#     def __init__(self, min_size, max_size=None):
#         self.min_size = min_size
#         if max_size is None:
#             max_size = min_size
#         self.max_size = max_size
#
#     def __call__(self, image, target):
#         size = random.randint(self.min_size, self.max_size)
#         # 这里size传入的是int类型，所以是将图像的最小边长缩放到size大小
#         image = F.resize(image, size)
#         # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
#         # 如果是之前的版本需要使用PIL.Image.NEAREST
#         # target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
#         target = F.resize(target, size, interpolation=Image.NEAREST)
#         return image, target

# 随机缩放
class RandomResize(object):
    def __init__(self, size, ratio=(0.5, 2.0)):
        self.size = size
        self.ratio = ratio

    def __call__(self, image, target):
        r = random.uniform(self.ratio[0], self.ratio[1])
        # size = int(self.size * r)
        size = int(self.size * r) if isinstance(self.size, int) else tuple(int(s * r) for s in self.size)
        image = F.resize(image, size)
        # 这里的interpolation注意下，在torchvision(0.9.0)以后才有InterpolationMode.NEAREST
        # 如果是之前的版本需要使用PIL.Image.NEAREST
        # target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        target = F.resize(target, size, interpolation=Image.NEAREST)
        return image, target


# 随机翻转
class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


# 随机裁剪
class RandomCrop(object):
    def __init__(self, size):
        if isinstance(size, int):
            size = (size, size)
        elif isinstance(size, list):
            size = tuple(size)
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size[0], self.size[1]))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


# 中心裁剪
class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


# 随机旋转
class RandomRotation(object):
    def __init__(self, rotation_prob, angle=15):
        self.rotation = rotation_prob
        self.angle = angle

    def __call__(self, image, target):
        if random.random() < self.rotation:
            angle = random.randint(-self.angle, self.angle)
            image = F.rotate(image, angle, expand=True, fill=0)
            target = F.rotate(target, angle, expand=True, fill=255)
            assert image.size == target.size
        return image, target


# 高斯模糊
class GaussianBlur(object):
    # 模糊半径越大, 正态分布标准差越大, 图像就越模糊
    """
    kernel_size：模糊半径。必须是奇数。
    sigma：正态分布的标准差。如果是浮点型，则固定；如果是二元组(min, max)，sigma在区间中随机选取一个值。
    """

    def __init__(self, kernel_size=11, sigma=2):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, image, target):
        aug = T.GaussianBlur(self.kernel_size, self.sigma)
        return aug(image), target


# 颜色扰动
class ColorJitter(object):
    # 随机改变图片的亮度，对比度和饱和度。
    """
    brightness：亮度；允许输入浮点型或二元组(min, max)。如果是浮点型，那么亮度在[max(0, 1 ## brightness), 1 + brightness]区间随机变换；如果是元组，亮度在给定的元组间随机变换。不允许输入负值。
    contrast：对比度。允许输入规则和亮度一致。
    saturation：饱和度。允许输入规则和亮度一致。
    hue：色调。允许输入浮点型或二元组(min, max)。如果是浮点型，那么亮度在[-hue, hue]区间随机变换；如果是元组，亮度在给定的元组间随机变换。不允许输入负值。必须满足0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5。

    """

    def __init__(self, p=(0.5, 1.5)):
        self.p = p

    def __call__(self, image, target):
        index = random.randint(0, 4)
        if index == 0:
            aug = T.ColorJitter(brightness=self.p)
        elif index == 1:
            aug = T.ColorJitter(contrast=self.p)
        elif index == 2:
            aug = T.ColorJitter(saturation=self.p)
        else:
            aug = T.ColorJitter(brightness=self.p, contrast=self.p, saturation=self.p)
        return aug(image), target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, ratio=(0.5, 2.0), hflip_prob=0.5, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        trans = [
            RandomResize(base_size, ratio),
            #ColorJitter(),
            #RandomRotation(rotation_prob=0.5),
            #GaussianBlur(),
        ]
        if hflip_prob > 0:
            trans.append(RandomHorizontalFlip(hflip_prob))
        trans.extend([
            RandomCrop(crop_size),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])
        self.transforms = Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = Compose([
            RandomResize(base_size, ratio=(1., 1.)),
            ToTensor(),
            Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def build_transform(cfg="config/example.yaml", train=True):
    if isinstance(cfg, dict):
        config = cfg
    else:
        import yaml
        yaml_file = Path(cfg).name
        with open(cfg) as f:
            config = yaml.safe_load(f)

    base_size = config["train"]["base_size"]
    crop_size = config["train"]["crop_size"]
    mean = tuple(config["train"]["mean"])
    std = tuple(config["train"]["std"])
    ratio = config["train"]["ratio"]

    return SegmentationPresetTrain(base_size, crop_size, ratio=ratio, mean=mean,
                                   std=std) if train else SegmentationPresetEval(
        base_size, mean=mean, std=std)


# if __name__ == '__main__':
#     mean = (0.485, 0.456, 0.406)
#     std = (0.229, 0.224, 0.225)
#     trans = [
#         RandomRotation(1.0),
#         RandomResize(size=200, ratio=(0.5, 2.0)),
#         RandomHorizontalFlip(1.0),
#         RandomCrop((224, 500)),
#         # GaussianBlur(11,2),
#         # ColorJitter(),
#         # ToTensor(),
#         # Normalize(mean=mean, std=std),
#     ]
#
#     transforms = Compose(trans)
#
#     from PIL import Image
#
#     img = Image.open(r'E:\pic\3.jpg')
#     for i in range(10):
#         a, b = transforms(img, img)
#         print(a.size)
#         import matplotlib.pyplot as plt
#
#         fig, ax = plt.subplots(1, 2)
#         ax[0].set_title('image')
#         ax[0].imshow(a)
#         ax[1].set_title(f'mask')
#         ax[1].imshow(b)
#         plt.xticks([]), plt.yticks([])
#         plt.show()
