import json
import os
import random

import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm


class SupConDatasetCifar10(torchvision.datasets.CIFAR10):
    def __init__(self, data_dir, train, transform, second_stage):
        super().__init__(root=data_dir, train=train, download=True, transform=transform)

        self.second_stage = second_stage
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.data[idx], self.targets[idx]

        # leave this part unchanged. The reason for this implementation - in the first stage of training
        # you have TwoCropTransform(actual transforms), so you have to call it by self.transform(img)
        # on the other hard, in the second stage of training there is no wrapper, so it's a regular
        # albumentation trans block, so it's called by self.transform(image=img)['image']
        if self.second_stage:
            image = self.transform(image=image)['image']
        else:
            image = self.transform(image)

        return image, label


class SupConDatasetCifar100(torchvision.datasets.CIFAR100):
    def __init__(self, data_dir, train, transform, second_stage):
        super().__init__(root=data_dir, train=train, download=True, transform=transform)

        self.second_stage = second_stage
        self.transform = transform

    def __getitem__(self, idx):
        image, label = self.data[idx], self.targets[idx]

        # leave this part unchanged. The reason for this implementation - in the first stage of training
        # you have TwoCropTransform(actual transforms), so you have to call it by self.transform(img)
        # on the other hard, in the second stage of training there is no wrapper, so it's a regular
        # albumentation trans block, so it's called by self.transform(image=img)['image']
        if self.second_stage:
            image = self.transform(image=image)['image']
        else:
            image = self.transform(image)

        return image, label


class SupConDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None, second_stage=True):
        super(SupConDataset, self).__init__()
        self.second_stage = second_stage
        self.transform = transform
        source = data_dir

        if train:
            self.data = source["train_images"]
            self.targets = source["train_labels"]
        else:
            self.data = source["val_images"]
            self.targets = source["val_labels"]

    def __getitem__(self, idx):
        images_path, label = self.data[idx], self.targets[idx]
        image = Image.open(images_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = np.asarray(image)

        # leave this part unchanged. The reason for this implementation - in the first stage of training
        # you have TwoCropTransform(actual transforms), so you have to call it by self.transform(img)
        # on the other hard, in the second stage of training there is no wrapper, so it's a regular
        # albumentation trans block, so it's called by self.transform(image=img)['image']
        if self.second_stage:
            image = self.transform(image=image)['image']
        else:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(batch):
        # ???????????????default_collate????????????
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


DATASETS = {'cifar10': SupConDatasetCifar10,
            'cifar100': SupConDatasetCifar100,
            'MyDataSet': SupConDataset}


def create_supcon_dataset(dataset_name, data_dir, train, transform, second_stage):  # , csv, second_stage):
    try:
        if os.path.exists(data_dir) and dataset_name not in ["cifar10", "cifar100"]:
            dataset_name = 'MyDataSet'
            train_images_path, val_images_path, train_images_label, val_images_label, every_class_num = read_split_data(
                data_dir, save_dir='./', val_rate=0.2, plot_image=False)
            data_dir = {"train_images": train_images_path,
                        "val_images": val_images_path,
                        "train_labels": train_images_label,
                        "val_labels": val_images_label}
        return DATASETS[dataset_name](data_dir, train, transform, second_stage)  # , csv, second_stage)
    except KeyError:
        Exception('Can\'t find such a dataset. Either use cifar10 or cifar100, or write your own one in datasets')


def read_split_data(data_root, save_dir, val_rate=0.2, plot_image=False):
    random.seed(0)  # ???????????????????????????
    assert os.path.exists(data_root), "data path:{} does not exists".format(data_root)

    # ???????????????????????????????????????????????????
    classes = [cla for cla in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, cla))]
    # ???????????????????????????
    classes.sort()

    # ?????????????????????????????????????????????
    class_indices = dict((k, v) for v, k in enumerate(classes))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open(save_dir + '/class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # ????????????????????????????????????
    train_images_label = []  # ???????????????????????????????????????
    val_images_path = []  # ????????????????????????????????????
    val_images_label = []  # ???????????????????????????????????????
    every_class_num = []  # ?????????????????????????????????
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # ???????????????????????????

    # ?????????????????????????????????
    for cla in tqdm(classes):
        cla_path = os.path.join(data_root, cla)
        # ????????????supported???????????????????????????
        images = [os.path.join(data_root, cla, i) for i in os.listdir(cla_path) if os.path.splitext(i)[-1] in supported]
        # ??????????????????????????????
        image_class = class_indices[cla]
        # ??????????????????????????????
        every_class_num.append(len(images))
        # ?????????????????????????????????
        val_path = random.sample(images, k=int(len(images) * val_rate))

        train_txt = open(save_dir + '/train.txt', 'w')
        val_txt = open(save_dir + '/val.txt', 'w')
        for img_path in images:
            if img_path in val_path:  # ???????????????????????????????????????????????????????????????
                val_txt.write(img_path + "\n")
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # ?????????????????????
                train_txt.write(img_path + "\n")
                train_images_path.append(img_path)
                train_images_label.append(image_class)
        train_txt.close()
        val_txt.close()

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    if plot_image:
        # ?????????????????????????????????
        plt.bar(range(len(classes)), every_class_num, align='center')
        # ????????????0,1,2,3,4??????????????????????????????
        plt.xticks(range(len(classes)), classes)
        # ?????????????????????????????????
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # ??????x??????
        plt.xlabel('image class')
        # ??????y??????
        plt.ylabel('number of images')
        # ????????????????????????
        plt.title('class distribution')
        plt.savefig(os.path.join(save_dir, "classes.jpg"))
        # plt.show()
    return train_images_path, val_images_path, train_images_label, val_images_label, every_class_num


if __name__ == '__main__':
    from transforms import build_transforms, TwoCropTransform

    data_Dir = r'E:\dataset\flow_data\train'
    transform = build_transforms(second_stage=False)
    data = create_supcon_dataset(dataset_name=None, data_dir=data_Dir, train=True,
                                 transform=transform['train_transforms'],
                                 second_stage=True)
    d = data[0]
    data1 = create_supcon_dataset(dataset_name=None, data_dir=data_Dir, train=True,
                                  transform=TwoCropTransform(transform['train_transforms']), second_stage=False)
    d1 = data1[0]
    print(d)
    print(d1)
