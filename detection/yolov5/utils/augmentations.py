# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Image augmentation functions
"""

import logging
import math
import random

import cv2
import numpy as np

from utils.general import colorstr, segment2box, resample_segments, check_version
from utils.metrics import bbox_ioa


class Albumentations:
    """使用Albumentations库增强数据"""
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self):
        self.transform = None
        try:
            import albumentations as A
            check_version(A.__version__, '1.0.3')  # version requirement

            self.transform = A.Compose([
                A.Blur(p=0.1),      # 随机的kernel大小模糊图片, p为概率
                A.MedianBlur(p=0.1),# 使用中值滤波模糊图片
                A.ToGray(p=0.01)],  # 转灰度图
                bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

            logging.info(colorstr('albumentations: ') + ', '.join(f'{x}' for x in self.transform.transforms if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            logging.info(colorstr('albumentations: ') + f'{e}')

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        return im, labels


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        # 随机取-1到1三个实数，乘以hyp中的hsv三通道的系数
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        # 分离通道
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        # 随机调整hsv之后重新组合通道
        # cv2.LUT(x, table)以x中的值为索引取根据table中的值
        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


def hist_equalize(im, clahe=True, bgr=False):
    """直方图均衡化"""
    # Equalize histogram on BGR image 'im' with im.shape(n,m,3) and range 0-255
    # BGR格式转化为YUV格式
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        # cv2.createCLAHE生成自适应均衡化图像
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        # 全局均衡化
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB


def replicate(im, labels):
    """随机偏移标签中心，生成新的标签与原标签结合"""
    # Replicate labels
    h, w = im.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = ((x2 - x1) + (y2 - y1)) / 2  # side length (pixels)
    # s.size返回ndarray的元素数量
    # 生成原标签个数一半的新标签
    for i in s.argsort()[:round(s.size * 0.5)]:  # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        # 随机偏移标签中心点
        yc, xc = int(random.uniform(0, h - bh)), int(random.uniform(0, w - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        im[y1a:y2a, x1a:x2a] = im[y1b:y2b, x1b:x2b]  # im4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return im, labels


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    # 计算缩放因子
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    """
    缩放(resize)到输入大小img_size的时候，如果没有设置上采样的话，则只进行下采样
    因为上采样图片会让图片模糊，对训练不友好影响性能。
    """
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    # 计算padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    # 获取最小的矩形填充
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    # 如果scaleFill=True,则不进行填充，直接resize成img_size,任由图片进行拉伸和压缩
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    # 计算上下左右填充大小
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # 进行填充
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def random_perspective(im, targets=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    # 设置平移仿射系数
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    # 设置透视变换矩阵
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    # 设置旋转和缩放的仿射矩阵
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    # 设置裁剪的仿射矩阵系数
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    # 融合仿射矩阵并作用在图片上
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im = cv2.warpPerspective(im, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:  # affine
            # 如不进行透视变换则只取M矩阵前两行，进行仿射变换
            im = cv2.warpAffine(im, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped

    # Transform label coordinates
    # 调整框的标签
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        # 如果使用segment标签
        if use_segments:  # warp segments
            # 先对segment标签进行重采样
            # 比如说segment坐标只有100个，通过interp函数将其采样为n个(默认1000)
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                # 对标签坐标进行透视/仿射变换
                xy = xy @ M.T  # transform
                xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]  # perspective rescale or affine

                # clip
                # 根据segment的坐标，取xy坐标的最大最小值，得到边框的坐标
                new[i] = segment2box(xy, width, height)

        else:  # warp boxes
            # 直接对box透视/仿射变换
            # 由于有旋转，透视变换等操作，所以需要对四个角点都进行变换
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            # 生成新标签
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return im, targets


def copy_paste(im, labels, segments, p=0.5):
    """通过segment标签粘贴目标到其他图片
    im:图片
    labels:[N, 5], cls, box:xyxy
    p:概率
    """
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    # p != 0且存在segment标签
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)
        # 从标签中随机选取p*n个目标
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            # box 为l标签框在im上水平镜像的框（y坐标不变，x坐标镜像）
            box = w - l[3], l[2], w - l[1], l[4]
            # 计算box与所有labels的iou，这个iou是交集与labels框面积的比值，而不是与并集的比值
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            # 如果box遮挡所有labels框的部分都不超过30%
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                # 添加box对应的labels，segments
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                # 画出该原mask的轮廓, 并用255填充mask区域
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (255, 255, 255), cv2.FILLED)

        # cv2.bitwise_and利用掩膜（mask）进行“与”操作，即掩膜图像白色区域是对需要处理图像像素的保留，
        # 黑色区域是对需要处理图像像素的剔除
        # 获得该mask在原图上的结果，即保留原图中mask的区域，其他的为黑
        result = cv2.bitwise_and(src1=im, src2=im_new)
        # 左右翻转的到需要贴到原图上的结果, 对应上面左右镜像得到的box
        result = cv2.flip(result, 1)  # augment segments (flip left-right)
        # 选取resule中mask区域的索引
        i = result > 0  # pixels to replace
        # i[:, :] = result.max(2).reshape(h, w, 1)  # act over ch
        # 再更新到原图上，就完成了将某个segment对象粘贴到它左右镜像的位置
        im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

    return im, labels, segments


def cutout(im, labels, p=0.5):
    """cutout数据增强, 给图片随机添加随机大小的方块噪声
    im:图片
    labels:[N, 5], cls, box:xyxy
    p:概率
    """
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    if random.random() < p:
        h, w = im.shape[:2]
        # 设置cutout添加噪声的scale
        scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16  # image size fraction
        for s in scales:
            mask_h = random.randint(1, int(h * s))  # create random masks
            mask_w = random.randint(1, int(w * s))

            # box
            xmin = max(0, random.randint(0, w) - mask_w // 2)
            ymin = max(0, random.randint(0, h) - mask_h // 2)
            xmax = min(w, xmin + mask_w)
            ymax = min(h, ymin + mask_h)

            # apply random color mask
            # 添加随机颜色的噪声
            im[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for _ in range(3)]

            # return unobscured labels
            if len(labels) and s > 0.03:
                box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
                # 计算噪声box与标签交集/标签的面积
                ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
                # 保留cutout噪声遮挡小于60%的标签
                labels = labels[ioa < 0.60]  # remove >60% obscured labels

    return labels


def mixup(im, labels, im2, labels2):
    """mixup数据增强, 按比例融合两张图片
    im:图片
    labels:[N, 5], cls, box:xyxy
    im2:图片
    labels2:[M, 5], cls, box:xyxy
    """
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    # 随机从beta分布中获取比例,range[0, 1]
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    # 按照比例融合两张图片
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    # 将两张图片标签拼接到一起
    labels = np.concatenate((labels, labels2), 0)
    return im, labels


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    """去除进行上面一系列操作后，被裁剪过小的框(面积小于裁剪前的area_thr)
       并且还有长和宽必须大于wh_thr个像素，且长宽比范围在(1/ar_thr, ar_thr)之间的限制"""
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates
