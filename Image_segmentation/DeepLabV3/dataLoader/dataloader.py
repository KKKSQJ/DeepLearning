import os
from pathlib import Path

from torch.utils.data import random_split

from dataLoader.voc_dataset import VOCSegmentation
from dataLoader.base_dataset import BasicDataset

import torch

DATASET = {
    "base": BasicDataset,
    "voc": VOCSegmentation
}


def build_dataloader(cfg='config/example.yaml', transform=None):
    if isinstance(cfg, dict):
        config = cfg
    else:
        import yaml
        yaml_file = Path(cfg).name
        with open(cfg) as f:
            config = yaml.safe_load(f)

    data_config = config["dataset"]
    dataset_type = data_config["type"]
    label_mapping = config["train"]["label_mapping"]
    # class_weights = config["train"]["class_weights"]
    assert str(dataset_type) in DATASET.keys(), f"datab type: '{dataset_type}' not in {DATASET.keys()}"

    if dataset_type == 'base':
        images_dir = data_config["train"]["images_path"]
        masks_dir = data_config["train"]["masks_path"]
        dataset = DATASET[dataset_type](images_dir, masks_dir, transforms=transform["train"],
                                        label_mapping=label_mapping)

        val_images_dir = data_config["val"]["images_path"]
        val_masks_dir = data_config["val"]["masks_path"]
        if (val_images_dir or val_masks_dir) is None:
            n_val = int(len(dataset) * 0.2)
            n_train = len(dataset) - n_val
            train_dataset, val_dataset = random_split(dataset, [n_train, n_val],
                                                      generator=torch.Generator().manual_seed(0))
        else:
            assert os.path.exists(val_images_dir), f"{val_images_dir} does not exists"
            assert os.path.exists(val_masks_dir), f"{val_masks_dir} does not exists"
            train_dataset = dataset
            val_dataset = DATASET[dataset_type](val_images_dir, val_masks_dir, transforms=transform["val"],
                                                label_mapping=label_mapping)


    elif dataset_type == "voc":
        voc_root = data_config["voc_root"]
        train_txt_name = "train.txt"
        train_dataset = DATASET[dataset_type](voc_root, transforms=transform["train"], txt_name=train_txt_name,
                                              label_mapping=label_mapping)

        val_txt_name = "val.txt"
        val_dataset = DATASET[dataset_type](voc_root, transforms=transform["val"], txt_name=val_txt_name,
                                            label_mapping=label_mapping)

    batch_size = config["train"]["batch_size"]
    num_workers = config["train"]["num_workers"]
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)

    return {"train": train_loader,
            "val": val_loader}

# if __name__ == '__main__':
#     import dataLoader.transforms as T
#     import matplotlib.pyplot as plt
#     from PIL import Image
#
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     trans = T.Compose(
#         [T.RandomResize(480, 480),
#          T.RandomCrop(480),
#          T.ToTensor(),
#          T.Normalize(mean=mean, std=std)
#          ]
#     )
#     loader = build_dataloader(cfg='../config/example.yaml', transform=trans)
#     train_loader = loader["train"]
#     val_loader = loader["val"]
#
#     pallette = []
#     color = {"0": [0, 0, 255], "1": [128, 0, 0], "255": [255, 255, 255]}
#     for i in color.values():
#         pallette += i
#
#     for i, data in enumerate(train_loader):
#         image = data[0][0].numpy()
#         mask = data[1][0].numpy()
#
#         image = image.transpose((1, 2, 0))
#         image = image * std + mean
#         image = image * 255.0
#         image = image.astype('uint8')[:, :, ::-1]
#         mask = mask.astype('uint8')
#         mask = Image.fromarray(mask)
#
#         plt.subplot(1, 2, 1)
#         plt.imshow(image)
#         plt.title("image")
#
#         plt.subplot(1, 2, 2)
#         mask.putpalette(pallette)
#         plt.imshow(mask)
#         plt.title("mask")
#         plt.ion()
#         plt.pause(0.01)
#         plt.waitforbuttonpress()
#         plt.show()
