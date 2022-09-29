import os
import cv2
import random
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import os.path

from PIL import Image
import torch
from torch.utils.data import Dataset


def read_split_data(data_root, save_dir, val_rate=0.2, plot_image=False):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(data_root), "data path:{} does not exists".format(data_root)

    # 遍历文件夹，一个文件夹对应一个类别
    classes = [cla for cla in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, cla))]
    # 排序，保证顺序一致
    classes.sort()

    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(classes))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open(save_dir + '/class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型

    # 遍历每个文件夹下的文件
    train_txt = open(save_dir + '/train.txt', 'w')
    val_txt = open(save_dir + '/val.txt', 'w')
    for cla in tqdm(classes):
        cla_path = os.path.join(data_root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(data_root, cla, i) for i in os.listdir(cla_path) if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_txt.write(img_path + " " + str(image_class) + "\n")
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_txt.write(img_path + " " + str(image_class) + "\n")
                train_images_path.append(img_path)
                train_images_label.append(image_class)
    train_txt.close()
    val_txt.close()

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(classes)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(classes)), classes)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('class distribution')
        plt.savefig(os.path.join(save_dir, "classes.jpg"))
        # plt.show()
    return train_images_path, val_images_path, train_images_label, val_images_label, every_class_num




class StandardDATASET(Dataset):
    """自定义数据集"""

    def __init__(self, images_path, images_class, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != "RGB":
            img = img.convert("RGB")
        label = self.images_class[item]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


class TxtDATASET(Dataset):
    """自定义数据集"""

    def __init__(self, txt_path, type="train.txt", transform=None):
        self.transform = transform
        with open(os.path.join(txt_path, type), 'r') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image_path = self.data[item].strip().split(" ")[0]
        img = Image.open(image_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        label = int(self.data[item].strip().split(" ")[-1])
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

if __name__ == '__main__':
    read_split_data(data_root=r'E:\dataset\classifition\flow_data\train', save_dir='./')
