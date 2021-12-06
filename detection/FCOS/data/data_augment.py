import torch
import math, random
from PIL import Image
import random
import torchvision.transforms as transforms


class Transforms(object):
    def __init__(self):
        pass

    def __call__(self, img, boxes):
        if random.random() < 0.3:
            img, boxes = colorJitter(img, boxes)
        if random.random() < 0.5:
            img, boxes = random_rotation(img, boxes)
        if random.random() < 0.5:
            img, boxes = random_crop_resize(img, boxes)
        return img, boxes


def colorJitter(img, boxes, brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1):
    img = transforms.ColorJitter(brightness=brightness,
                                 contrast=contrast, saturation=saturation, hue=hue)(img)
    return img, boxes


def random_rotation(img, boxes, degree=10):
    d = random.uniform(-degree, degree)
    w, h = img.size
    rx0, ry0 = w / 2.0, h / 2.0
    img = img.rotate(d)
    a = -d / 180.0 * math.pi
    boxes = torch.from_numpy(boxes)
    new_boxes = torch.zeros_like(boxes)
    new_boxes[:, 0] = boxes[:, 1]
    new_boxes[:, 1] = boxes[:, 0]
    new_boxes[:, 2] = boxes[:, 3]
    new_boxes[:, 3] = boxes[:, 2]
    for i in range(boxes.shape[0]):
        ymin, xmin, ymax, xmax = new_boxes[i, :]
        xmin, ymin, xmax, ymax = float(xmin), float(ymin), float(xmax), float(ymax)
        x0, y0 = xmin, ymin
        x1, y1 = xmin, ymax
        x2, y2 = xmax, ymin
        x3, y3 = xmax, ymax
        z = torch.FloatTensor([[y0, x0], [y1, x1], [y2, x2], [y3, x3]])
        tp = torch.zeros_like(z)
        tp[:, 1] = (z[:, 1] - rx0) * math.cos(a) - (z[:, 0] - ry0) * math.sin(a) + rx0
        tp[:, 0] = (z[:, 1] - rx0) * math.sin(a) + (z[:, 0] - ry0) * math.cos(a) + ry0
        ymax, xmax = torch.max(tp, dim=0)[0]
        ymin, xmin = torch.min(tp, dim=0)[0]
        new_boxes[i] = torch.stack([ymin, xmin, ymax, xmax])
    new_boxes[:, 1::2].clamp_(min=0, max=w - 1)
    new_boxes[:, 0::2].clamp_(min=0, max=h - 1)
    boxes[:, 0] = new_boxes[:, 1]
    boxes[:, 1] = new_boxes[:, 0]
    boxes[:, 2] = new_boxes[:, 3]
    boxes[:, 3] = new_boxes[:, 2]
    boxes = boxes.numpy()
    return img, boxes


def _box_inter(box1, box2):
    tl = torch.max(box1[:, None, :2], box2[:, :2])  # [n,m,2]
    br = torch.min(box1[:, None, 2:], box2[:, 2:])  # [n,m,2]
    hw = (br - tl).clamp(min=0)  # [n,m,2]
    inter = hw[:, :, 0] * hw[:, :, 1]  # [n,m]
    return inter


def random_crop_resize(img, boxes, crop_scale_min=0.2, aspect_ratio=[3. / 4, 4. / 3], remain_min=0.7, attempt_max=10):
    success = False
    boxes = torch.from_numpy(boxes)
    for attempt in range(attempt_max):
        # choose crop size
        area = img.size[0] * img.size[1]
        target_area = random.uniform(crop_scale_min, 1.0) * area
        aspect_ratio_ = random.uniform(aspect_ratio[0], aspect_ratio[1])
        w = int(round(math.sqrt(target_area * aspect_ratio_)))
        h = int(round(math.sqrt(target_area / aspect_ratio_)))
        if random.random() < 0.5:
            w, h = h, w
        # if size is right then random crop
        if w <= img.size[0] and h <= img.size[1]:
            x = random.randint(0, img.size[0] - w)
            y = random.randint(0, img.size[1] - h)
            # check
            crop_box = torch.FloatTensor([[x, y, x + w, y + h]])
            inter = _box_inter(crop_box, boxes)  # [1,N] N can be zero
            box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])  # [N]
            mask = inter > 0.0001  # [1,N] N can be zero
            inter = inter[mask]  # [1,S] S can be zero
            box_area = box_area[mask.view(-1)]  # [S]
            box_remain = inter.view(-1) / box_area  # [S]
            if box_remain.shape[0] != 0:
                if bool(torch.min(box_remain > remain_min)):
                    success = True
                    break
            else:
                success = True
                break
    if success:
        img = img.crop((x, y, x + w, y + h))
        boxes -= torch.Tensor([x, y, x, y])
        boxes[:, 1::2].clamp_(min=0, max=h - 1)
        boxes[:, 0::2].clamp_(min=0, max=w - 1)
        # ow, oh = (size, size)
        # sw = float(ow) / img.size[0]
        # sh = float(oh) / img.size[1]
        # img = img.resize((ow,oh), Image.BILINEAR)
        # boxes *= torch.FloatTensor([sw,sh,sw,sh])
    boxes = boxes.numpy()
    return img, boxes


def flip(img, boxes):
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    w = img.width
    if boxes.shape[0] != 0:
        xmin = w - boxes[:, 2]
        xmax = w - boxes[:, 0]
        boxes[:, 2] = xmax
        boxes[:, 0] = xmin
    return img, boxes


if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(0)
    img = Image.open('d:/pic/2.jpg')
    boxes = [10, 10, 50, 50]
    boxes = np.array(boxes, dtype=np.float32)[None, :]
    if not isinstance(boxes.dtype, np.float32):
        boxes = np.array(boxes, dtype=np.float32)
    plt.imshow(img)

    plt.figure(1)
    imgf, boxesf = flip(img, boxes)
    imgc, boxesc = colorJitter(img, boxes)
    imgr, boxesr = random_rotation(img, boxes)
    imgs, boxess = random_crop_resize(img, boxes)
    plt.imshow(imgs)
    # plt.ioff()
    plt.show()
