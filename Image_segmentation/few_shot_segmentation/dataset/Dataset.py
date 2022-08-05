r""" Dataloader builder for few-shot semantic segmentation dataset  """
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset.pascal import DatasetPASCAL
from dataset.coco import DatasetCOCO
from dataset.fss import DatasetFSS
from dataset.fewshot import DatasetFewShot


# from dataset.mars import DatasetMARS


class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize):
        cls.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
            'fss': DatasetFSS,
            'fewshot':DatasetFewShot
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        if isinstance(img_size, int):
            img_size = (img_size, img_size)

        cls.transform = transforms.Compose([transforms.Resize(size=(img_size[0], img_size[1])),
                                            transforms.ToTensor(),
                                            transforms.Normalize(cls.img_mean, cls.img_std)])

    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1,nclass=None):
        # Force randomness during training for diverse episode combinations
        # Freeze randomness during testing for reproducibility
        shuffle = split == 'trn'
        nworker = nworker if split == 'trn' else 0

        dataset = cls.datasets[benchmark](cls.datapath, fold=fold, transform=cls.transform, split=split, shot=shot,
                                          use_original_imgsize=cls.use_original_imgsize,nclass=nclass)
        dataloader = DataLoader(dataset, batch_size=bsz, shuffle=shuffle, num_workers=nworker)

        return dataloader




