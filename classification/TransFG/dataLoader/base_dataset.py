from torch.utils.data import Dataset
import torch

import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, ids=None, images=None, labels=None, transforms=None):
        super(BasicDataset, self).__init__()
        if labels is None:
            labels = []
        if images is None:
            images = []
        if ids is None:
            ids = []

        self.transforms = transforms
        self.ids = ids
        self.images = images
        self.labels = labels
        assert len(self.images) == len(self.labels)
        logging.info(f"====>  Creating dataset with {len(self.ids)} examples   <=====")
        if len(images) == 0:
            raise RuntimeError(f"====>  Dataset is empty ,Please check !!!   <=====")

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image label.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target = self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
