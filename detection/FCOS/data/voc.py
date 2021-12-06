#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import glob
import os
import os.path
import pickle
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from loguru import logger

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from .data_augment import Transforms, flip

from .voc_classes import VOC_CLASSES

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes


class AnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES)))
        )
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0, 5))
        for obj in target.iter("object"):
            difficult = obj.find("difficult")
            if difficult is not None:
                difficult = int(difficult.text) == 1
            else:
                difficult = False
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.strip()
            bbox = obj.find("bndbox")

            pts = ["xmin", "ymin", "xmax", "ymax"]
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                # cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        width = int(target.find("size").find("width").text)
        height = int(target.find("size").find("height").text)
        img_info = (height, width)

        return res, img_info


class VOCDetection(Dataset):
    """
    VOC Detection Dataset Object

    input is image, target is annotation

    Args:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(
            self,
            data_dir,
            image_sets=None,
            img_size=(416, 416),
            preproc=None,
            target_transform=AnnotationTransform(),
            dataset_name="VOC0712",
            is_train=True,
    ):
        super().__init__()
        if image_sets is None:
            image_sets = [("2007", "trainval")]
        self.root = os.path.join(data_dir, "VOCdevkit")
        self.image_set = image_sets
        self.img_size = img_size
        self.preproc = preproc
        self.is_train = is_train
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join("%s", "Annotations", "%s.xml")
        self._imgpath = os.path.join("%s", "JPEGImages", "%s.jpg")
        self._classes = VOC_CLASSES
        self.ids = list()
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        for (year, name) in image_sets:
            self._year = year
            rootpath = os.path.join(self.root, "VOC" + year)
            for line in open(
                    os.path.join(rootpath, "ImageSets", "Main", name + ".txt")
            ):
                self.ids.append((rootpath, line.strip()))

        self.annotations = self._load_coco_annotations()

        self.imgs = None

    def __len__(self):
        return len(self.ids)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in range(len(self.ids))]

    def load_anno_from_ids(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()
        assert self.target_transform is not None
        res, img_info = self.target_transform(target)
        height, width = img_info
        boxes = res[:, :4]
        classes = res[:, 4]

        return (boxes, classes, img_info)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_image(self, index):
        img_id = self.ids[index]
        img = Image.open(self._imgpath % img_id)
        if img.mode != "RGB":
            img = img.convert("RGB")
        assert img is not None

        return img

    def preprocess_img_boxes(self, image, boxes, input_ksize):
        """

        :param image: []
        :param boxes: [None,4]
        :param input_ksize: image_paded
        :return:
        """

        min_side, max_side = input_ksize
        h, w, _ = image.shape

        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        pad_w = 32 - nw % 32
        pad_h = 32 - nh % 32

        image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return image_paded, boxes

    def pull_item(self, index):
        """Returns the original image and target at an index for mixup

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            img, target
        """

        if self.imgs is None:
            img = self.load_image(index)
        else:
            img = self.imgs[index]
        boxes, classes, img_info = self.annotations[index]

        if not isinstance(boxes.dtype, np.float32):
            boxes = np.array(boxes, dtype=np.float32)

        if self.is_train:
            if random.random() < 0.5:
                img, boxes = flip(img, boxes)
            if self.preproc is not None:
                img, boxes = self.preproc(img, boxes)

        return img, boxes, classes, img_info, index

    def __getitem__(self, index):
        img, boxes, classes, img_info, img_id = self.pull_item(index)

        img = np.array(img)
        boxes = np.array(boxes, dtype=np.float32)
        classes = np.array(classes)
        if self.img_size is not None:
            img, boxes = self.preprocess_img_boxes(img, boxes, self.img_size)
        img = transforms.ToTensor()(img)
        boxes = torch.from_numpy(boxes)
        classes = torch.LongTensor(classes)

        return img, boxes, classes

    def collate_fn(self, data):
        imgs_list, boxes_list, classes_list = zip(*data)
        assert len(imgs_list) == len(boxes_list) == len(classes_list)
        batch_size = len(boxes_list)
        pad_imgs_list = []
        pad_boxes_list = []
        pad_classes_list = []

        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        for i in range(batch_size):
            img = imgs_list[i]
            pad_imgs_list.append(transforms.Normalize(self.mean, self.std, inplace=True)(
                torch.nn.functional.pad(img, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.)))

        max_num = 0
        for i in range(batch_size):
            n = boxes_list[i].shape[0]
            if n > max_num: max_num = n
        for i in range(batch_size):
            pad_boxes_list.append(
                torch.nn.functional.pad(boxes_list[i], (0, 0, 0, max_num - boxes_list[i].shape[0]), value=-1))
            pad_classes_list.append(
                torch.nn.functional.pad(classes_list[i], (0, max_num - classes_list[i].shape[0]), value=-1))

        batch_boxes = torch.stack(pad_boxes_list)
        batch_classes = torch.stack(pad_classes_list)
        batch_imgs = torch.stack(pad_imgs_list)

        return batch_imgs, batch_boxes, batch_classes

    @property
    def classes(self):
        return self._classes

    @property
    def id2name(self):
        name2id = dict(zip(self._classes, range(len(self._classes))))
        _id2name = {v: k for k, v in name2id.items()}
        return _id2name


if __name__ == '__main__':
    # path = r'D:\pic\2.jpg'
    # img = cv2.imread(path)
    # img1 = Image.open(path)
    # img1 = np.array(img1)
    # img2 = img1[:, :, ::-1]
    # # img2 = np.ascontiguousarray(img2)
    # # img2 = img2.astype('float32')
    # cv2.imshow('a', img)
    # cv2.imshow('b', img1)
    # cv2.imshow('c', img2)
    # cv2.waitKey()
    # image_resized = cv2.resize(img1, (100, 100))
    # print(1)

    train_dataset = VOCDetection(
        data_dir=r'D:\dataset\VOC2012',
        image_sets=[("2012", "trainval")],
        img_size=(416, 416),
        preproc=Transforms(),
        target_transform=AnnotationTransform(),
        dataset_name="VOC0712",
        is_train=True,
    )
    print(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=8,
                                               shuffle=True,
                                               collate_fn=train_dataset.collate_fn,
                                               num_workers=0, worker_init_fn=np.random.seed(0))
    # for i,data in enumerate(train_dataset):
    #     print(data)
    dataiter = iter(train_loader)
    img, boxes, classes = next(dataiter)
    print(img, boxes, classes)
