import cv2
import numpy as np
import random
from dataLoader.data_augment import *


def transform_train(image, mask):
    add_ = 0
    image = cv2.resize(image, (256, 256))
    mask = cv2.resize(mask, (256, 256))
    mask = mask[:, :, None]

    image = np.concatenate([image, mask], 2)

    if 1:
        if random.random() < 0.5:
            image = np.fliplr(image)
            # 15587指的是类别数量，这里将翻转后的图片视为一个新的类别label
            add_ += 15587
        image, mask = image[:, :, :3], image[:, :, 3]
    if random.random() < 0.5:
        image, mask = random_angle_rotate(image, mask, angles=(-25, 25))
    # noise
    if random.random() < 0.5:
        index = random.randint(0, 1)
        if index == 0:
            image = do_speckle_noise(image, sigma=0.1)
        elif index == 1:
            image = do_gaussian_noise(image, sigma=0.1)
    if random.random() < 0.5:
        index = random.randint(0, 3)
        if index == 0:
            image = do_brightness_shift(image, 0.1)
        elif index == 1:
            image = do_gamma(image, 1)
        elif index == 2:
            image = do_clahe(image)
        elif index == 3:
            image = do_brightness_multiply(image)
    if 1:
        image, mask = random_erase(image, mask, p=0.5)
    if 1:
        image, mask = random_shift(image, mask, p=0.5)
    if 1:
        image, mask = random_scale(image, mask, p=0.5)
    # todo data augment
    if 1:
        if random.random() < 0.5:
            mask[...] = 0
    mask = mask[:, :, None]
    # image = np.concatenate([image, mask], 2)
    image = np.transpose(image, (2, 0, 1))
    image = image.copy().astype(np.float)
    image = torch.from_numpy(image).div(255).float()
    return image,add_


def transform_valid(image, mask):
    images = []
    image = cv2.resize(image, (256, 256))
    mask = cv2.resize(mask, (256, 256))
    mask = mask[:, :, None]
    # image = np.concatenate([image, mask], 2)
    raw_image = image.copy()

    image = np.transpose(raw_image, (2, 0, 1))
    image = image.copy().astype(np.float)
    image = torch.from_numpy(image).div(255).float()
    images.append(image)

    image = np.fliplr(raw_image)
    image = np.transpose(image, (2, 0, 1))
    image = image.copy().astype(np.float)
    image = torch.from_numpy(image).div(255).float()
    images.append(image)
    return images


def train_collate(batch):
    batch_size = len(batch)
    images = []
    labels = []
    for b in range(batch_size):
        if batch[b][0] is None:
            continue
        else:
            images.extend(batch[b][0])
            labels.extend(batch[b][1])
    images = torch.stack(images, 0)
    labels = torch.from_numpy(np.array(labels))
    return images, labels


def valid_collate(batch):
    batch_size = len(batch)
    images = []
    labels = []
    names = []
    for b in range(batch_size):
        if batch[b][0] is None:
            continue
        else:
            images.extend(batch[b][0])
            labels.append(batch[b][1])
            names.append(batch[b][2])
    images = torch.stack(images, 0)
    labels = torch.from_numpy(np.array(labels))
    return images, labels, names
