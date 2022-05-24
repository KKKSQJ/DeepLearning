import logging
import os.path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from models.transfg import VisionTransformer
from dataLoader.dataset import CUB, Cars, dogs, NASBirds, INat2017, MyDataset
from dataLoader.autoaugment import AutoAugImageNetPolicy
from utils.schuduler import WarmupCosineSchedule, WarmupLinearSchedule
from losses.contrastive_loss import contrastive_loss
from losses.labelSmoothing import LabelSmoothing

from torchvision import transforms
import torch
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import torchvision
import torch.optim as optim
import torch_optimizer as jettify_optim

# model_name: model
MODEL = {"ViT-B_16": VisionTransformer,
         "ViT-B_32": VisionTransformer,
         "ViT-L_16": VisionTransformer,
         "ViT-L_32": VisionTransformer,
         "ViT-H_14": VisionTransformer,
         "Testing": VisionTransformer}

# dataset_name: dataset
DATASET = {"CUB_200_2011": CUB,
           "car": Cars,
           "nabirds": NASBirds,
           "dog": dogs,
           "INat2017": INat2017,
           "MySet": MyDataset,
           }

# transfrom_name: transforms
TRANSFORMS = {"Resize": transforms.Resize,
              "RandomCrop": transforms.RandomCrop,
              "CenterCrop": transforms.CenterCrop,
              "AutoAugImageNetPolicy": AutoAugImageNetPolicy,
              "RandomHorizontalFlip": transforms.RandomHorizontalFlip,
              "ToTensor": transforms.ToTensor,
              "Normalize": transforms.Normalize}

# optimizer_name: optimizer
OPTIMIZERS = {
    "Adam": optim.Adam,
    'AdamW': optim.AdamW,
    "SGD": optim.SGD,
    'LookAhead': jettify_optim.Lookahead,
    'Ranger': jettify_optim.Ranger,
    'RAdam': jettify_optim.RAdam,
}

# scheduler_name: scheduler
SCHEDULERS = {
    "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
    "LambdaLR": optim.lr_scheduler.LambdaLR,
    "WarmupCosine": WarmupCosineSchedule,
    "WarmupLinear": WarmupLinearSchedule,
}

# loss_name: loss
LOSSES = {
    'CrossEntropy': torch.nn.CrossEntropyLoss,
    'LabelSmoothing': LabelSmoothing,
    'Contrastive': contrastive_loss,
}


def build_optim(params_to_optimize, optimizer_params=None, loss_params=None, scheduler_params=None):
    if loss_params is not None:
        if 'params' in loss_params:
            weight = loss_params['params']['weight']
            if weight is not None:
                loss_params["params"]["weight"] = torch.FloatTensor(loss_params['params']['weight']).cuda()
            criterion = LOSSES[loss_params['name']](**loss_params['params'])
        else:
            criterion = LOSSES[loss_params['name']]()
    else:
        criterion = None

    if optimizer_params is not None:
        optimizer = OPTIMIZERS[optimizer_params["name"]](params_to_optimize, **optimizer_params["params"])
    else:
        optimizer = None

    if scheduler_params:
        scheduler = SCHEDULERS[scheduler_params["name"]](optimizer, **scheduler_params["params"])
    else:
        scheduler = None

    return {"criterion": criterion, "optimizer": optimizer, "scheduler": scheduler}


def build_model(cfg='config/example.yaml', num_classes=20):
    if isinstance(cfg, dict):
        config = cfg
    else:
        import yaml
        with open(cfg, encoding='utf-8') as f:
            config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    assert model_name in MODEL.keys(), f"model name must in {MODEL.keys()}, but now model name is {model_name}"
    if model_name == 'ViT-B_16':
        config["model"]["patches"]["hidden_size"] = 768
        config["model"]["patches"]["patch_size"] = 16
        config["model"]["transformer"]["mlp_dim"] = 3072
        config["model"]["transformer"]["num_heads"] = 12
        config["model"]["transformer"]["num_layers"] = 12
    elif model_name == 'ViT-B_32':
        config["model"]["patches"]["hidden_size"] = 768
        config["model"]["patches"]["patch_size"] = 32
        config["model"]["transformer"]["mlp_dim"] = 3072
        config["model"]["transformer"]["num_heads"] = 12
        config["model"]["transformer"]["num_layers"] = 12
    elif model_name == 'ViT-L_16':
        config["model"]["patches"]["hidden_size"] = 1024
        config["model"]["patches"]["patch_size"] = 16
        config["model"]["transformer"]["mlp_dim"] = 4096
        config["model"]["transformer"]["num_heads"] = 16
        config["model"]["transformer"]["num_layers"] = 24
    elif model_name == 'ViT-L_32':
        config["model"]["patches"]["hidden_size"] = 1024
        config["model"]["patches"]["patch_size"] = 32
        config["model"]["transformer"]["mlp_dim"] = 4096
        config["model"]["transformer"]["num_heads"] = 16
        config["model"]["transformer"]["num_layers"] = 24
    elif model_name == 'ViT-H_14':
        config["model"]["patches"]["hidden_size"] = 1280
        config["model"]["patches"]["patch_size"] = 14
        config["model"]["transformer"]["mlp_dim"] = 5120
        config["model"]["transformer"]["num_heads"] = 16
        config["model"]["transformer"]["num_layers"] = 32
    elif model_name == 'Testing':
        config["model"]["patches"]["hidden_size"] = 1
        config["model"]["patches"]["patch_size"] = 16
        config["model"]["transformer"]["mlp_dim"] = 1
        config["model"]["transformer"]["num_heads"] = 1
        config["model"]["transformer"]["num_layers"] = 1

    model = MODEL[model_name](config, num_classes=num_classes)

    np_weights = config["train"]["np_weights"]
    if np_weights is not None:
        if os.path.exists(np_weights):
            model.load_from(np.load(np_weights))
            logging.info(f"===>  Load pretrain weights from {np_weights}  <====")
    return model


def build_transforms(cfg='config/example.yaml', dataset="CUB"):
    if isinstance(cfg, dict):
        config = cfg
    else:
        import yaml
        with open(cfg, encoding='utf-8') as f:
            config = yaml.safe_load(f)

    train_transforms = config["train_transforms"]
    val_transforms = config["val_transforms"]

    train_trans = []
    val_trans = []
    for key, value in train_transforms.items():
        if value is None:
            train_trans.append(TRANSFORMS[key]())
        elif key == 'Normalize':
            train_trans.append(TRANSFORMS[key](value[0], value[1]))
        else:
            train_trans.append(TRANSFORMS[key](tuple(value)))
    for key, value in val_transforms.items():
        if value is None:
            val_trans.append(TRANSFORMS[key]())
        elif key == 'Normalize':
            val_trans.append(TRANSFORMS[key](value[0], value[1]))
        else:
            val_trans.append(TRANSFORMS[key](tuple(value)))
    train_transform = transforms.Compose(train_trans)
    val_transform = transforms.Compose(val_trans)

    # assert dataset in DATASET.keys(), f"dataset must in {DATASET.keys()}, but now dataset is {dataset}"
    #
    # if dataset == 'CUB_200_2011':
    #     train_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
    #                                           transforms.RandomCrop((448, 448)),
    #                                           transforms.RandomHorizontalFlip(),
    #                                           transforms.ToTensor(),
    #                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #     val_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
    #                                         transforms.CenterCrop((448, 448)),
    #                                         transforms.ToTensor(),
    #                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # elif dataset == 'car':
    #     train_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
    #                                           transforms.RandomCrop((448, 448)),
    #                                           transforms.RandomHorizontalFlip(),
    #                                           AutoAugImageNetPolicy(),
    #                                           transforms.ToTensor(),
    #                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #     val_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
    #                                         transforms.CenterCrop((448, 448)),
    #                                         transforms.ToTensor(),
    #                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # elif dataset == 'dog':
    #     train_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
    #                                           transforms.RandomCrop((448, 448)),
    #                                           transforms.RandomHorizontalFlip(),
    #                                           transforms.ToTensor(),
    #                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #     val_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
    #                                         transforms.CenterCrop((448, 448)),
    #                                         transforms.ToTensor(),
    #                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # elif dataset == 'nabirds':
    #     train_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
    #                                           transforms.RandomCrop((448, 448)),
    #                                           transforms.RandomHorizontalFlip(),
    #                                           transforms.ToTensor(),
    #                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #     val_transform = transforms.Compose([transforms.Resize((600, 600), Image.BILINEAR),
    #                                         transforms.CenterCrop((448, 448)),
    #                                         transforms.ToTensor(),
    #                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # elif dataset == 'INat2017':
    #     train_transform = transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
    #                                           transforms.RandomCrop((304, 304)),
    #                                           transforms.RandomHorizontalFlip(),
    #                                           AutoAugImageNetPolicy(),
    #                                           transforms.ToTensor(),
    #                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #     val_transform = transforms.Compose([transforms.Resize((400, 400), Image.BILINEAR),
    #                                         transforms.CenterCrop((304, 304)),
    #                                         transforms.ToTensor(),
    #                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    return train_transform, val_transform


def build_loader(cfg='config/example.yaml', train_transforms=None, val_transforms=None):
    if isinstance(cfg, dict):
        config = cfg
    else:
        import yaml
        with open(cfg, encoding='utf-8') as f:
            config = yaml.safe_load(f)

    if config["train"]["local_rank"] not in [-1, 0]:
        torch.distributed.barrier()

    dataset_name = config["dataset"]["name"]
    data_len = config["train"]["data_len"]

    assert dataset_name in DATASET.keys(), f"dataset name must in {DATASET.keys()}, but now dataset name is {dataset_name}"
    if dataset_name == 'CUB_200_2011':
        root = config["dataset"]["root"]
        trainset = DATASET[dataset_name](root=root, data_len=data_len, train=True, transforms=train_transforms)
        valset = DATASET[dataset_name](root=root, data_len=data_len, train=False, transforms=val_transforms)
    elif dataset_name == 'car':
        root = config["dataset"]["root"]
        mat_anno = os.path.join(root, 'devkit/cars_train_annos.mat')
        train_data_dir = os.path.join(root, 'cars_train')
        val_data_dir = os.path.join(root, 'cars_test')
        car_names = os.path.join(root, 'devkit/cars_meta.mat')
        cleaned = os.path.join(root, 'cleaned.dat')
        trainset = DATASET[dataset_name](mat_anno=mat_anno, data_dir=train_data_dir, car_names=car_names,
                                         data_len=data_len, cleaned=None, transforms=train_transforms)
        valset = DATASET[dataset_name](mat_anno=mat_anno, data_dir=val_data_dir, car_names=car_names, data_len=data_len,
                                       cleaned=None, transforms=val_transforms)
    elif dataset_name == 'dog':
        root = config["dataset"]["root"]
        trainset = DATASET[dataset_name](root=root, train=True, cropped=False, data_len=data_len,
                                         transforms=train_transforms,
                                         download=True)
        valset = DATASET[dataset_name](root=root, train=False, cropped=False, data_len=data_len,
                                       transforms=val_transforms, download=True)

    elif dataset_name == 'nabirds':
        root = config["dataset"]["root"]
        trainset = DATASET[dataset_name](root=root, train=True, transform=train_transforms)
        valset = DATASET[dataset_name](root=root, train=False, transform=val_transforms)

    elif dataset_name == 'INat2017':
        root = config["dataset"]["root"]
        trainset = DATASET[dataset_name](root=root, split='train', transform=train_transforms)
        valset = DATASET[dataset_name](root=root, split='val', transform=val_transforms)

    batch_size = config["train"]["batch_size"]
    nw = config["train"]["num_workers"]
    if config["train"]["local_rank"] not in [-1, 0]:
        torch.distributed.barrier()

    train_sampler = RandomSampler(trainset) if config["train"]["local_rank"] == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(valset) if config["train"]["local_rank"] == -1 else DistributedSampler(valset)
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              num_workers=nw,
                              drop_last=True,
                              pin_memory=True)
    val_loader = DataLoader(valset,
                            sampler=test_sampler,
                            batch_size=batch_size,
                            num_workers=nw,
                            pin_memory=True) if valset is not None else None
    return train_loader, val_loader


def display_dataset(data_loader):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    n_row = 8
    padding = 50
    pad_value = 10

    for i, data in enumerate(data_loader):
        fig = plt.figure(figsize=(15, 10))
        img_grid = torchvision.utils.make_grid(data[0], nrow=n_row, padding=padding, pad_value=pad_value)
        img_grid = img_grid.numpy().transpose((1, 2, 0))
        img_grid = (img_grid * std + mean) * 255.0
        img_grid = img_grid.astype('uint8')[..., ::-1]

        plt.imshow(img_grid)
        for index, label in enumerate(data[1]):
            plt.text(index % n_row * (data[0][0].shape[2] + padding) + padding,
                     index // 8 * (data[0][0].shape[1] + 2 * padding),
                     "label:" + str(label.item()))
        plt.ion()
        plt.pause(1)
        # plt.waitforbuttonpress()
        plt.close()
        plt.show()

        # image = data[0][0].numpy()
        # label = int(data[1][0])
        # image = image.transpose((1, 2, 0))
        # image = image * std + mean
        # image = image * 255.0
        # image = image.astype('uint8')[:, :, ::-1]
        # plt.imshow(img_grid)
        # plt.title("label:" + str(label))
        # plt.ion()
        # plt.pause(0.01)
        # plt.waitforbuttonpress()
        # plt.show()


if __name__ == '__main__':
    cfg = '../config/example.yaml'
    model = build_model(cfg=cfg, num_classes=20)
    x = torch.randn(16, 3, 448, 448)
    y = model(x)
    print(y)
    print(model)

    train_trans, val_trans = build_transforms(cfg=cfg)
    print(train_trans)

    train_loader, val_loader = build_loader(cfg=cfg, train_transforms=train_trans, val_transforms=val_trans)
    print(train_loader)

    display_dataset(train_loader)
