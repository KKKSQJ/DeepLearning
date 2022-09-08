import os
import copy
import random

import torch
import numpy as np
from PIL import Image
import cv2
from pycocotools.coco import COCO
from torch.utils.data import Dataset
import json

from tqdm import tqdm


class Keypoint(Dataset):
    def __init__(self, img_path, anno_path, dataset_path=None, transforms=None):
        super(Keypoint, self).__init__()

        assert os.path.exists(img_path), f"{img_path} does not exists!"
        assert os.path.exists(anno_path), f"{anno_path} does not exists!"

        self.transforms = transforms
        self.img_path = img_path
        self.anno_path = anno_path
        # self.max_objs = 100  # 每张图上最多点个数
        # self.down_ratio = 4  # 下采样倍数
        # self.regs = np.zeros((self.max_objs, 2), dtype=np.float32)      # 每个点偏移量
        # self.inds = np.zeros((self.max_objs,), dtype=np.int64)
        # self.ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)     # 点的mask

        if dataset_path is None:
            ids = [x.split('.')[0] for x in os.listdir(anno_path) if x.endswith('.json')]
        else:
            assert os.path.exists(dataset_path), f"{dataset_path} dose not exists!"
            with open(dataset_path, 'r') as f:
                ids = f.read().splitlines()

        self.ids = ids
        self.info_list = []

        for id in self.ids:
            info = {
                "image_id": id,
                "image_path": os.path.join(img_path, id + ".jpg"),
                "json_path": os.path.join(anno_path, id + ".json")
            }
            json_path = os.path.join(anno_path, id + ".json")
            with open(json_path, 'r', encoding='utf-8') as f:
                img_info = json.load(f)

            info["image_width"] = img_info['imageWidth']
            info["image_height"] = img_info['imageHeight']

            keypoints = []
            labels = []
            for shape in img_info['shapes']:
                # if shape['label'] == '0':
                #     continue
                label = [0] * len(shape['points']) if int(shape['label']) == 0 else [1] * len(shape['points']) * int(
                    shape['label'])
                labels.extend(label)
                keypoints.append(np.array(shape["points"]))

            keypoints = np.concatenate(keypoints, axis=0)
            info["keypoints"] = np.array(keypoints)
            info["labels"] = np.array(labels)
            info["visible"] = np.ones((keypoints.shape[0], 1))
            self.info_list.append(info)

        # print(1)

    def __getitem__(self, idx):
        target = copy.deepcopy(self.info_list[idx])
        image = cv2.imread(target["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image, info = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.info_list)

    @staticmethod
    def collate_fn(batch):
        imgs_tuple, targets_tuple = tuple(zip(*batch))
        imgs_tensor = torch.stack(imgs_tuple)
        return imgs_tensor, targets_tuple


def read_split_data(data_root, save_dir, val_rate=0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(data_root), "data path:{} does not exists".format(data_root)
    os.makedirs(save_dir, exist_ok=True)

    files = [file for file in os.listdir(data_root) if file.endswith(".json")]
    files.sort()
    random.shuffle(files)

    f = open(save_dir + '/train.txt', 'w')
    f1 = open(save_dir + '/val.txt', 'w')

    for i, id in enumerate(files):
        if i < int(len(files) * val_rate):
            f1.write(id.strip().split('.')[0] + "\n")
        else:
            f.write(id.strip().split('.')[0] + "\n")

    f.close()
    f1.close()


if __name__ == '__main__':
    dataset = Keypoint(img_path=r'E:\dataset\pose_estimation\insulator\images',
                       anno_path=r'E:\dataset\pose_estimation\insulator\annos')

    # read_split_data(r'E:\dataset\pose_estimation\Sum_7.31_120\sum_json', '../data')

    from dataset.kp_transforms import *

    for image, info in dataset:
        print(image)
        new_image, new_info = AffineTransform(scale=(0.65, 1.35), rotation=(-45, 45), fixed_size=(448, 448))(image,
                                                                                                             info)
        # new_image, new_info = AffineTransform(fixed_size=(512, 512))(image,info)
        #
        new_image, new_info = RandomHorizontalFlip(1)(new_image, new_info)
        new_image, new_info = KeypointToHeatMap((448 // 4, 448 // 4), keypoints_nums=1, gaussian_sigma=2)(new_image,
                                                                                                          new_info)

        import matplotlib.pyplot as plt
        from utils.draw_utils import draw_keypoints
        from torchvision import transforms

        fig = plt.figure()
        resize_img = draw_keypoints(new_image, new_info["keypoints"])
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(resize_img)
        for i, heatmap in enumerate(new_info["heatmap"]):
            img = transforms.ToPILImage()(heatmap)
            ax = fig.add_subplot(1, 2, 2)
            ax.imshow(img)
        plt.show()

        print(info)
    # pass
