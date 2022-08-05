import numpy as np
from PIL import Image, ImageOps
import random
import torch
from torchvision import transforms
from torchvision.transforms import functional as F

def crop(img, mask, size):
    # padding height or width if smaller than cropping size
    if isinstance(size, int):
        size = (size, size)
    w, h = img.size
    padw = size[0] - w if w < size[0] else 0
    padh = size[1] - h if h < size[1] else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)

    # cropping
    w, h = img.size
    x = random.randint(0, w - size[0])
    y = random.randint(0, h - size[1])
    img = img.crop((x, y, x + size[0], y + size[1]))
    mask = mask.crop((x, y, x + size[0], y + size[1]))

    return img, mask


def hflip(img, mask):
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


# 随机缩放
def resize(img,mask,size,ratio=(0.5,2.0)):
    r = random.uniform(ratio[0], ratio[1])
    size = int(size * r) if isinstance(size, int) else tuple(int(s * r) for s in size)
    image = F.resize(img, size)
    target = F.resize(mask, size, interpolation=Image.NEAREST)
    return image, target


def normalize(img, mask):
    """
    :param img: PIL image
    :param mask: PIL image, corresponding mask
    :return: normalized torch tensor of image and mask
    """
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    mask = torch.from_numpy(np.array(mask)).long()
    return img, mask
