import os

import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import random


class VOCSegmentation(data.Dataset):
    def __init__(self, voc_root, year="2012", transforms=None, txt_name: str = "train.txt"):
        super(VOCSegmentation, self).__init__()
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClass')

        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    # 计算该batch数据中，channel, h, w的最大值
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


class Happy_Whale_Dataset(data.Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None):
        super(Happy_Whale_Dataset, self).__init__()
        if isinstance(image_dir or mask_dir, list):
            self.masks = mask_dir
            self.images = image_dir
        else:
            assert os.path.exists(image_dir)
            assert os.path.exists(mask_dir)

            mask_files = os.listdir(mask_dir)
            self.masks = [os.path.join(mask_dir, x) for x in mask_files]
            self.images = [os.path.join(image_dir, x) for x in mask_files if os.path.exists(os.path.join(image_dir, x))]

        assert len(self.masks) == len(self.images)
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        # target = np.array(target)
        # target[target < 175] = 0
        # target[target >= 175] = 1

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=0)
        return batched_imgs, batched_targets


def split_train_eval(image_dir, mask_dir, eval_ratio=0.1):
    random.seed(100)
    assert os.path.exists(image_dir)
    assert os.path.exists(mask_dir)
    assert 0.0 <= eval_ratio <= 1.0
    mask_file = os.listdir(mask_dir)
    random.shuffle(mask_file)
    data = {}
    train_image = []
    train_mask = []
    eval_image = []
    eval_mask = []

    for i in range(len(mask_file)):
        if i < int(len(mask_file) * eval_ratio):
            eval_image.append(os.path.join(image_dir, mask_file[i]))
            eval_mask.append(os.path.join(mask_dir, mask_file[i]))
        else:
            train_image.append(os.path.join(image_dir, mask_file[i]))
            train_mask.append(os.path.join(mask_dir, mask_file[i]))
    train = [train_image, train_mask]
    eval = [eval_image, eval_mask]
    data["train"] = train
    data["eval"] = eval
    return train, eval


if __name__ == '__main__':
    import transforms as T


    class SegmentationPresetTrain:
        def __init__(self, base_size, crop_size, hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
            min_size = int(0.5 * base_size)
            max_size = int(2.0 * base_size)

            trans = [T.RandomResize(min_size, max_size)]
            if hflip_prob > 0:
                trans.append(T.RandomHorizontalFlip(hflip_prob))
            trans.extend([
                T.RandomCrop(crop_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
            self.transforms = T.Compose(trans)

        def __call__(self, img, target):
            return self.transforms(img, target)


    # dataset = VOCSegmentation(voc_root=r"E:\dataset\VOC2012",
    #                           transforms=SegmentationPresetTrain(base_size=520, crop_size=480))
    # d1 = dataset[0]
    # print(d1)

    dataset = Happy_Whale_Dataset(image_dir=r'E:\dataset\humpback-whale-identification\train',
                                  mask_dir='E:\dataset\humpback-whale-identification\masks')
    d1 = dataset[0]
    plt.figure(0)
    plt.imshow(d1[0])
    plt.figure(1)
    plt.imshow(d1[1])
    plt.show()
    print(d1)
