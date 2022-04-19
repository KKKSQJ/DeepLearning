import os
import numpy as np
import cv2
import torch

from dataLoader.base_dataset import BaseDataset

SUPPORT_IMG = [".jpg", ".png", ".jpeg"]


class My_DataSet(BaseDataset):
    def __init__(self,
                 img_path=None,
                 label_path=None,
                 num_sample=None,
                 multi_scale=True,
                 flip=True,
                 brightness=True,
                 mode="train",
                 ignore_label=-1,
                 base_size=2048,
                 crop_size=(512, 1024),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]
                 ):
        super(My_DataSet, self).__init__(ignore_label, base_size,
                                         crop_size, downsample_rate, scale_factor, mean, std, )

        self.img_path = img_path
        self.label_path = label_path

        self.multi_scale = multi_scale
        self.flip = flip
        self.brightness = brightness
        self.mode = mode

        self.file_names = []
        self.imgs_list = []
        self.label_list = []

        self.read_imgs()
        self.read_labels()

        if mode == 'train':
            assert len(self.imgs_list) == len(self.label_list) == len(self.file_names)
        else:
            assert len(self.imgs_list) == len(self.file_names)

        self.files = self.read_files(self.imgs_list, self.label_list, self.file_names)
        if num_sample:
            self.files = self.files[:num_sample]

        # self.label_mapping = {-1: ignore_label, 0: ignore_label,
        #                       1: ignore_label, 2: ignore_label,
        #                       3: ignore_label, 4: ignore_label,
        #                       5: ignore_label, 6: ignore_label,
        #                       7: 0, 8: 1, 9: ignore_label,
        #                       10: ignore_label, 11: 2, 12: 3,
        #                       13: 4, 14: ignore_label, 15: ignore_label,
        #                       16: ignore_label, 17: 5, 18: ignore_label,
        #                       19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
        #                       25: 12, 26: 13, 27: 14, 28: 15,
        #                       29: ignore_label, 30: ignore_label,
        #                       31: 16, 32: 17, 33: 18}

        self.label_mapping = {255: ignore_label, 0: 0, 1: 1}
        self.class_weights = torch.FloatTensor([1.0, 1.0]).cuda()

        # self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345,
        #                                         1.0166, 0.9969, 0.9754, 1.0489,
        #                                         0.8786, 1.0023, 0.9539, 0.9843,
        #                                         1.1116, 0.9037, 1.0865, 1.0955,
        #                                         1.0865, 1.1529, 1.0507]).cuda()

    def read_imgs(self):
        if isinstance(self.img_path, list):
            for file in self.img_path:
                _file = file.split(os.sep)[-1]
                name, p = os.path.splitext(_file)
                if p.lower() in SUPPORT_IMG:
                    self.file_names.append(name)
                    self.imgs_list.append(file)
        else:
            assert os.path.exists(self.img_path), f"{self.img_path} does not exists!"

            for file in os.listdir(self.img_path):
                name, p = os.path.splitext(file)
                if p.lower() in SUPPORT_IMG:
                    self.file_names.append(name)
                    self.imgs_list.append(os.path.join(self.img_path, file))

    def read_labels(self):
        if self.mode != 'train':
            return
        assert os.path.exists(self.label_path), f"{self.label_path} does not exists!"
        if len(self.file_names):
            for name in self.file_names:
                self.label_list.append(os.path.join(self.label_path, name + ".png"))

    def read_files(self, imgs, labels, names):
        files = []
        if self.mode == 'train':
            for (img, label, name) in zip(imgs, labels, names):
                files.append({"img": img,
                              "label": label,
                              "name": name})
        else:
            for (img, name) in zip(imgs, names):
                files.append({"img": img,
                              "name": name})
        return files

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(item["img"], cv2.IMREAD_COLOR)
        size = image.shape
        if self.mode == 'train':
            label = cv2.imread(item["label"], cv2.IMREAD_GRAYSCALE)
            label = self.convert_label(label)
            image, label = self.gen_sample(image, label, self.multi_scale, self.flip, self.brightness)
            return image.copy(), label.copy(), np.array(size), name
        else:
            if self.downsample_rate != 1:
                image = cv2.resize(image, None, fx=self.downsample_rate, fy=self.downsample_rate,
                                   interpolation=cv2.INTER_NEAREST)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name


if __name__ == '__main__':
    data = My_DataSet(img_path=r'E:\dataset\car\train_images', label_path=r'E:\dataset\car\train_masks')
    d = data[0]
    print(d)
