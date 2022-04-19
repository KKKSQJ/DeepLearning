import os
from pathlib import Path

import cv2

from dataLoader.base_dataset import BaseDataset
from dataLoader.my_dataset import My_DataSet

from torch.utils.data import DataLoader, random_split
import torch


def build_loaders(cfg, mode="train"):
    if isinstance(cfg, dict):
        config = cfg
    else:
        import yaml
        yaml_file = Path(cfg).name
        with open(cfg) as f:
            config = yaml.safe_load(f)

    data_config = config["dataset"]

    if mode == "train":
        train_image_path = data_config["train"]["image_path"]
        train_label_path = data_config["train"]["label_path"]

        val_image_path = data_config["val"]["image_path"]
        val_label_path = data_config["val"]["label_path"]

        assert os.path.exists(train_label_path), f"{train_image_path} does not exists"
        assert os.path.exists(train_label_path), f"{train_label_path} does not exists"

        train_config = config["train"]
        base_size = train_config["base_size"]
        crop_size = train_config["image_size"]
        num_sample = train_config["num_sample"]
        multi_scale = train_config["multi_scale"]
        flip = train_config["flip"]
        brightness = train_config["brightness"]
        downsample_rate = train_config["downsample_rate"]
        scale_factor = train_config["scale_factor"]

        train_dataset = My_DataSet(
            img_path=train_image_path,
            label_path=train_label_path,
            num_sample=num_sample,
            multi_scale=multi_scale,
            flip=flip,
            brightness=brightness,
            mode="train",
            ignore_label=-1,
            base_size=base_size,
            crop_size=crop_size,
            downsample_rate=downsample_rate,
            scale_factor=scale_factor,
        )

        if (val_label_path or val_image_path) is None:
            n_val = int(len(train_dataset) * 0.2)
            n_train = len(train_dataset) - n_val
            train_dataset, val_dataset = random_split(train_dataset, [n_train, n_val],
                                                      generator=torch.Generator().manual_seed(0))
        else:
            assert os.path.exists(val_image_path), f"{val_image_path} does not exists"
            assert os.path.exists(val_label_path), f"{val_label_path} does not exists"

            val_dataset = My_DataSet(
                img_path=val_image_path,
                label_path=val_label_path,
                num_sample=num_sample,
                multi_scale=multi_scale,
                flip=flip,
                brightness=brightness,
                mode="train",
                ignore_label=-1,
                base_size=base_size,
                crop_size=crop_size,
                downsample_rate=downsample_rate,
                scale_factor=scale_factor,
            )

        batch_size = train_config["batch_size"]
        num_workers = train_config["num_workers"]

        loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
        val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)
        return {"train_loader": train_loader,
                "val_loader": val_loader,
                "train_dataset": train_dataset,
                "val_dataset": val_dataset}

    elif mode == 'test':
        test_image_path = data_config["test"]["image_path"]

        assert os.path.exists(test_image_path), f"{test_image_path} does not exists"

        test_config = config["test"]
        base_size = test_config["base_size"]
        crop_size = test_config["image_size"]
        num_sample = test_config["num_sample"]
        multi_scale = test_config["multi_scale"]
        flip = test_config["flip"]
        brightness = test_config["brightness"]
        downsample_rate = test_config["downsample_rate"]

        test_dataset = My_DataSet(
            img_path=test_image_path,
            label_path=None,
            num_sample=num_sample,
            multi_scale=multi_scale,
            flip=flip,
            brightness=brightness,
            mode="test",
            ignore_label=-1,
            base_size=base_size,
            crop_size=crop_size,
            downsample_rate=downsample_rate,
        )

        batch_size = test_config["batch_size"]
        num_workers = test_config["num_workers"]

        loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

        return {"test_loader": test_loader,
                "test_Datset": test_dataset}


if __name__ == '__main__':
    import numpy as np

    loader = build_loaders(cfg='../config/example.yaml', mode='train')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for index, data in enumerate(loader["train_loader"]):
        img, label, _, _ = data
        img = img.detach().cpu().numpy()
        label = label.detach().cpu().numpy()
        for index, image in enumerate(img):
            image = image.transpose((1, 2, 0))
            image = image * std + mean
            image = image * 255.0
            image = image.astype('uint8')[:, :, ::-1]
            mask = label[index] * 255.0
            mask = mask.astype('uint8')
            cv2.imshow("image", image)
            cv2.imshow("mask", mask)
            cv2.waitKey()

    print(1)
