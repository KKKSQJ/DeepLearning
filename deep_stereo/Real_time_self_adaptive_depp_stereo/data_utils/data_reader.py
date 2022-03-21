import os
import re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from data_utils import preprocessing


def read_list_file(path_file):
    """
    Read dataset description file encoded as left;right;disp;conf
    Args:
        path_file: path to the file encoding the database
    Returns:
        [left,right,gt,conf] 4 list containing the images to be loaded
    """
    with open(path_file, 'r') as f_in:
        lines = f_in.readlines()
    lines = [x for x in lines if not (x.strip() == '' or x.strip()[0] == '#')]
    left_file_list = []
    right_file_list = []
    gt_file_list = []
    conf_file_list = []
    for l in lines:
        # to_load = re.split(',|;', l.strip())
        to_load = l.strip().split(" ")
        left_file_list.append(to_load[0])
        right_file_list.append(to_load[1])
        if len(to_load) > 2:
            gt_file_list.append(to_load[2])
        if len(to_load) > 3:
            conf_file_list.append(to_load[3])
    return left_file_list, right_file_list, gt_file_list, conf_file_list


def read_image_from_path(image_patth):
    image = Image.open(image_patth)
    if image.mode != 'RGB':
        image = image.convert("RGB")
    image = np.array(image)
    return image


def read_pfm_from_path(pfm_path):
    """
    Load a pfm file as a numpy array
    Args:
        file: path to the file to be loaded
    Returns:
        content of the file as a numpy array
    """
    color = None
    width = None
    height = None
    scale = None
    endian = None
    file = open(pfm_path, 'rb')
    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dims = file.readline()
    try:
        width, height = list(map(int, dims.split()))
    except:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width, 1)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale




class dataset(Dataset):
    """
    Class that reads a dataset for deep stereo
    """

    def __init__(self,
                 path_file,
                 crop_shape=[320, 1216],
                 augment=False,
                 is_training=True,
                 transform = None
                 ):
        super(dataset, self).__init__()
        if not os.path.exists(path_file):
            raise Exception('File not found during dataset construction')

        self._path_file = path_file
        self._crop_shape = crop_shape
        self._augment = augment
        self._is_training = is_training
        self._transform = transform
        self._build_input_pipeline()

    def _build_input_pipeline(self):
        # path_file:txt,每行格式：左图路径，右图路径 ，视差图路径
        left_files, right_files, gt_files, _ = read_list_file(self._path_file)
        # 一组图片组成一个couples [左图，右图，视差图]
        self._couples = [[l, r, gt] for l, r, gt in zip(left_files, right_files, gt_files)]
        # 是否使用pfm，视差图
        self._usePfm = gt_files[0].endswith('pfm') or gt_files[0].endswith('PFM')
        if not self._usePfm:
            gg = cv2.imread(gt_files[0], -1)
            self._double_prec_gt = (gg.dtype == np.uint16)

    def __len__(self):
        return len(self._couples)

    def __getitem__(self, item):
        couples = self._couples[item]
        left_file_path = couples[0]
        right_file_path = couples[1]
        gt_file_path = couples[2]

        left_image = read_image_from_path(left_file_path)
        right_image = read_image_from_path(right_file_path)
        if self._usePfm:
            gt_image, _ = read_pfm_from_path(gt_file_path)
        else:
            read_type = torch.int16 if self._double_prec_gt else torch.int8
            gt_image = read_image_from_path(gt_file_path)
            gt_image.dtype('float32')
            if self._double_prec_gt:
                gt_image = gt_image / 256.0

        # 裁剪GT的尺寸，与image保持一致
        gt_image = gt_image[:, :left_image.shape[1], :]

        if self._is_training:
            assert left_image.shape[:2] == right_image.shape[:2] == gt_image.shape[:2]
            left_image, right_image, gt_image = preprocessing.random_crop([left_image, right_image, gt_image],
                                                                          self._crop_shape)

        else:
            (left_image, right_image, gt_image) = [
                preprocessing.resize_image_with_crop_or_pad(x, self._crop_shape[0], self._crop_shape[1]) for x in
                [left_image, right_image, gt_image]]

        gt_image = np.where(np.isinf(gt_image),np.full_like(gt_image,0),gt_image)
        # if self._augment:
        #     left_image, right_image = preprocessing.augment(left_image, right_image)
        if self._transform is not None:
            left_image = self._transform(left_image)
            right_image = self._transform(right_image)
            gt_image = self._transform(gt_image)
        else:
            left_image = torch.from_numpy(left_image.transpose((2, 0, 1))).contiguous().float()
            right_image = torch.from_numpy(right_image.transpose((2, 0, 1))).contiguous().float()
            gt_image = torch.from_numpy(gt_image.transpose((2, 0, 1))).contiguous().float()

        return [left_image, right_image, gt_image]

def get_path_file(path=r'E:\dataset\deep_stereo\data'):
    f = open('file_path.txt','w')
    assert os.path.exists(path)
    for i in os.listdir(path):
        left_path = os.path.join(path,i,"im0.png")
        right_path = os.path.join(path,i,"im1.png")
        gt_path = os.path.join(path,i,"disp1.pfm")
        f.write("{} {} {}\n".format(left_path,right_path,gt_path))
    f.close()

if __name__ == '__main__':
    # get_path_file()
    path_file = 'file_path.txt'
    data = dataset(path_file=path_file)
    left,right,img = data.__getitem__(0)
    print(1)
