import json

from torch.utils.data import Dataset
import random
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def do_length_decode(rle, H=192, W=384, fill_value=255):
    mask = np.zeros((H, W), np.uint8)
    if type(rle).__name__ == 'float': return mask
    mask = mask.reshape(-1)
    rle = np.array([int(s) for s in rle.split(' ')]).reshape(-1, 2)
    for r in rle:
        start = r[0] - 1
        end = start + r[1]
        mask[start: end] = fill_value
    mask = mask.reshape(W, H).T  # H, W need to swap as transposing.
    return mask


class WhaleDataset(Dataset):
    def __init__(self,
                 data: dict,
                 class_id_label_file,
                 mode='train',
                 transform=None,
                 min_num_classes=0):
        super(WhaleDataset, self).__init__()

        self.names = data["name"]
        self.id = data["id"]
        self.image_path = data["img_path"]
        self.mask_path = data["mask_path"]
        self.mode = mode
        self.transform = transform
        self.min_num_classes = min_num_classes
        if isinstance(class_id_label_file, dict):
            self.labels = class_id_label_file
        else:
            self.labels = json.loads('./data/class_indices.json')

        # self.labels = self.id2label(self.id)
        self.filename_to_id = {Image: Id for Image, Id in zip(self.names, self.id)}

        if mode in ['train', 'valid']:
            self.dict_train = self.balance_train()
            # self.labels = list(self.dict_train.keys())
            self.ids = [k for k in self.dict_train.keys() if len(self.dict_train[k]) >= min_num_classes]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        names = self.dict_train[id]
        nums = len(names)

        if nums == 1:
            anchor_name = names[0]
            positive_name = names[0]
        else:
            anchor_name, positive_name = random.sample(names, 2)
        negative_label = random.choice(list(set(self.ids) ^ set([id])))
        negative_name = random.choice(self.dict_train[negative_label])

        anchor_image = self.get_image(anchor_name, self.transform)
        positive_image = self.get_image(positive_name, self.transform)
        negative_image = self.get_image(negative_name, self.transform)

        anchor_label = self.labels[id]
        positive_label = self.labels[id]
        negative_label = self.labels[negative_label]

        assert anchor_name != negative_name

        return [anchor_image, positive_image, negative_image], [anchor_label, positive_label, negative_label]

    def id2label(self, names):
        dict_label = {}
        id = 0
        for name in names:
            dict_label[name] = id
            id += 1
        return dict_label

    # id对应的图片列表。如 id:"2dsad0"，对应图片列表[xx.jpg,xxx.jpg]
    def balance_train(self):
        dict_train = {}
        for name, id in zip(self.names, self.id):
            if not id in dict_train.keys():
                dict_train[id] = [name]
            else:
                dict_train[id].append(name)
        return dict_train

    def get_image(self, name, transform):
        image = cv2.imread(os.path.join(self.image_path, "{}").format(name))
        if image is None:
            # TODO
            print('image is None {}'.format(os.path.join(self.image_path, name)))
        mask = cv2.imread(os.path.join(self.mask_path, "{}").format(name))
        if mask is None:
            mask = np.zeros_like(image[:, :, 0])

        image = transform(image, mask)
        return image


class WhaleTestDataset(Dataset):
    def __init__(self,
                 data: dict,
                 class_id_label_file,
                 mode='test',
                 transform=None,
                 ):
        super(WhaleTestDataset, self).__init__()

        self.names = data["name"]
        self.ids = data["id"]
        self.image_path = data["img_path"]
        self.mask_path = data["mask_path"]
        self.mode = mode
        self.transform = transform
        if isinstance(class_id_label_file,dict):
            self.labels = class_id_label_file
        else:
            self.labels = json.loads('./data/class_indices.json')

        #self.labels = self.id2label(self.ids)
        self.filename_to_id = {Image: Id for Image, Id in zip(self.names, self.ids)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if self.mode in ["test"]:
            name = self.names[index]
            image = self.get_image(name, self.transform)
            return image
        elif self.mode in ["valid", "train"]:
            name = self.names[index]
            label = self.labels[self.ids[index]]
            image = self.get_image(name, self.transform)
            return image, label, name

    def id2label(self, names):
        dict_label = {}
        id = 0
        for name in names:
            dict_label[name] = id
            id += 1
        return dict_label

    def get_image(self, name, transform):
        image = cv2.imread(os.path.join(self.image_path, "{}").format(name))
        if image is None:
            # TODO
            print('image is None {}'.format(os.path.join(self.image_path,name)))
        mask = cv2.imread(os.path.join(self.mask_path, "{}").format(name))
        if mask is None:
            mask = np.zeros_like(image[:, :, 0])

        image = transform(image, mask)
        return image
