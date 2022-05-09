import os

import numpy
import numpy as np
import torch
from PIL import Image
import logging

from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self,
                 images_dir,
                 masks_dir,
                 file_names=None,
                 transforms=None,
                 label_mapping=None,
                 ):
        if file_names is None:
            assert os.path.exists(images_dir), f"{images_dir} does not exists"
            assert os.path.exists(masks_dir), f"{masks_dir} does not exists"
            self.ids = [os.path.splitext(file)[0] for file in os.listdir(images_dir) if not file.startswith('.')]
            if not self.ids:
                raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        else:
            assert isinstance(file_names, list), "file_names must be a list,which store the file name"
            self.ids = file_names
        logging.info(f"====>  Creating dataset with {len(self.ids)} examples   <=====")

        self.images = [os.path.join(images_dir, x + ".jpg") for x in self.ids]
        self.masks = [os.path.join(masks_dir, x + ".png") for x in self.ids]
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms

        self.label_mapping = label_mapping
        # self.class_weights = torch.FloatTensor(class_weights).cuda()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])
        if self.label_mapping is not None:
            target = self.convert_label(target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def convert_label(self, mask, inverse=False):
        mask = np.asarray(mask).copy()
        temp = mask.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                mask[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                mask[temp == k] = v
        return Image.fromarray(mask)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))

        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.size() for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
