import os
import os.path
import sys
import time
from pathlib import Path
import argparse

import numpy as np

from utils import select_device
import yaml
import logging
import tempfile
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import torchvision
# from torchvision import transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset, distributed, dataset, dataloader
import torch.backends.cudnn as cudnn

from utils import increment_path, torch_distributed_zero_first, save_checkpoint, KpLoss, Kploss_focal, key_point_eval, \
    train_one_epoch, _reg_loss

from dataset import Keypoint, CocoKeypoint, read_split_data
from dataset import kp_transforms as transforms
from models import HighResolution as hrnet

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', -1))


# print("LOCAL RANK:", LOCAL_RANK)
# print("RANK:", RANK)
# print("WORLD_SIZE:", WORLD_SIZE)


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config/train.yaml')
    parser.add_argument('--arch', default='hrnet', help='model name')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--anno_path', type=str)
    parser.add_argument('--use_offset', action='store_true',default=False)
    parser.add_argument('--freeze_layers', action='store_true',default=False)
    parser.add_argument('--syncBN', action='store_true',default=False)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--start_epochs', type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", type=str, help="device = 'cpu' or '0' or '0,1,2,3'")
    parser.add_argument('--lr', type=float)
    parser.add_argument("--lrf", type=float)
    parser.add_argument("--print_freq", type=int)
    parser.add_argument('--weights', type=str)
    parser.add_argument('--resume', type=str)

    args = parser.parse_args()

    with open(vars(args)["config_path"], "r", encoding='utf-8') as config_file:
        config = yaml.full_load(config_file)

    if args.img_path is not None:
        config["data"]["img_path"] = args.img_path
    if args.anno_path is not None:
        config["data"]["anno_path"] = args.anno_path
    if args.arch is not None:
        config["train"]["arch"] = args.arch
    if args.use_offset is not None:
        config["train"]["use_offset"] = args.use_offset
    if args.print_freq is not None:
        config["train"]["print_freq"] = args.print_freq
    if args.resume is not None:
        config["train"]["resume"] = args.resume
    if args.device is not None:
        config["train"]["device"] = args.device
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.weights is not None:
        config["train"]["weights"] = args.weights
    if args.freeze_layers is not None:
        config["train"]["freeze_layers"] = args.freeze_layers
    if args.lr is not None:
        config["train"]["lr"] = args.lr
    if args.lrf is not None:
        config["train"]["lrf"] = args.lrf
    if args.start_epochs is not None:
        config["train"]["start_epochs"] = args.start_epochs
    if args.epochs is not None:
        config["train"]["epochs"] = args.epochs
    return config


def main(config):
    print(config)

    # Create logging
    if RANK in {-1, 0}:
        save_dir = Path(str(increment_path(Path(config["train"]["project"]) / config["train"]["name"],
                                           exist_ok=config["train"]["exist_ok"])))
        save_dir.mkdir(parents=True, exist_ok=True)

        weights_dir = save_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        tb_writer = SummaryWriter(str(save_dir))

        with open(save_dir / 'train.yaml', 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False)

        logging_path = str(save_dir / "train.log")
        logging.basicConfig(filename=logging_path, level=logging.INFO, filemode="w+")
        for k, v in config.items():
            logging.info(f"====>  {k}: {v}   <=====")

    # Init device
    device = select_device(config["train"]["device"], config["train"]["batch_size"])
    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        print("torch.cuda.device_count():", torch.cuda.device_count())
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    cuda = device.type != 'cpu'

    # Create data
    scale = (config["train"]["scale"][0], config["train"]["scale"][1])
    rotation = (config["train"]["rotation"][0], config["train"]["rotation"][1])
    fixed_size = (config["train"]["resolution"][0], config["train"]["resolution"][1])
    heatmap_hw = (fixed_size[0] // 4, fixed_size[1] // 4)
    data_transform = {
        "train": transforms.Compose(
            [transforms.AffineTransform(scale=scale, rotation=rotation, fixed_size=fixed_size),
             transforms.RandomHorizontalFlip(p=0.5),
             transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=1,
                                          keypoints_nums=config["train"]["num_joint"]),
             transforms.ToTensor(),
             transforms.Normalize([0.616, 0.231, 0.393], [0.312, 0.288, 0.250])]
            # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ),
        "val": transforms.Compose(
            [transforms.AffineTransform(scale=(1.0, 1.0), fixed_size=fixed_size),
             transforms.KeypointToHeatMap(heatmap_hw=heatmap_hw, gaussian_sigma=1,
                                          keypoints_nums=config["train"]["num_joint"]),
             transforms.ToTensor(),
             transforms.Normalize([0.616, 0.231, 0.393], [0.312, 0.288, 0.250])]
            # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )
    }

    img_path = config["data"]["img_path"]
    anno_path = config["data"]["anno_path"]
    if RANK in {-1, 0}:
        read_split_data(anno_path, save_dir='data', val_rate=0.2)

    with torch_distributed_zero_first(LOCAL_RANK):
        train_set = Keypoint(img_path=img_path, anno_path=anno_path, dataset_path="data/train.txt",
                             transforms=data_transform["train"])
        test_set = Keypoint(img_path=img_path, anno_path=anno_path, dataset_path="data/val.txt",
                            transforms=data_transform["val"])
        # train_set.data = train_set.data[:1000]
        # test_set.data = test_set.data[:1000]

    train_sampler = None if LOCAL_RANK == -1 else distributed.DistributedSampler(train_set, shuffle=True)
    test_sampler = None if LOCAL_RANK == -1 else distributed.DistributedSampler(test_set, shuffle=False)
    print(train_sampler)

    nw = min([os.cpu_count(), config["train"]["batch_size"] if config["train"]["batch_size"] > 1 else 0, 8])
    train_dataloader = DataLoader(train_set, batch_size=config["train"]["batch_size"], shuffle=train_sampler is None,
                                  num_workers=nw, sampler=train_sampler, pin_memory=True, drop_last=True,
                                  collate_fn=train_set.collate_fn)

    test_dataloader = DataLoader(test_set, batch_size=config["train"]["batch_size"], shuffle=False,
                                 num_workers=nw, sampler=test_sampler, pin_memory=True, drop_last=False,
                                 collate_fn=test_set.collate_fn)

    # Write to tensorboard
    if RANK in {-1, 0}:
        for i in range(10):
            img, target = test_set[i]
            tb_writer.add_image("test_set", img)
            tb_writer.add_image("heatmap", target['heatmap'])

    # Create model
    checkpoint_path = ""
    model = hrnet(base_channel=config["train"]["base_channel"], num_joint=config["train"]["num_joint"],
                  use_offset=config["train"]["use_offset"])

    # pretrain weights
    if config["train"]["weights"] is not None and os.path.exists(config["train"]["weights"]):
        with torch_distributed_zero_first(LOCAL_RANK):
            checkpoint_path = config["train"]["weights"]

        ckpt = torch.load(checkpoint_path, map_location='cpu')
        # ckpt = {k: v for k, v in ckpt.items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(ckpt, strict=False)
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")

        if RANK == 0:
            torch.save(model.state_dict(), checkpoint_path)
        if RANK != -1:
            dist.barrier()
            model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))

    model = model.to(device)

    # Freeze weights
    if config["train"]["freeze_layers"]:
        for name, p in model.named_parameters():
            if "fc" not in name:
                p.requires_grad_(False)
                logging.info("Freeze layer:{}".format(name))

    # 只有训练带有BN结构的网络时使用SyncBatchNorm才有意义，上面固定权重，使得BN层没有训练
    if config["train"]["syncBN"] and cuda and RANK != -1:
        # 使用SyncBatchNorm后训练会更耗时
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        logging.info('Using SyncBatchNorm()')

    # Convert to DDP
    if cuda and RANK != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # Optimizer
    # 学习率要根据并行GPU的数量进行倍增
    lr = config["train"]["lr"] * WORLD_SIZE if RANK != -1 else config["train"]["lr"]
    pg = [p for p in model.parameters() if p.requires_grad]

    # optimizer = torch.optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5e-4)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # lf = lambda x: ((1 + math.cos(x * math.pi / config["train"]["epochs"])) / 2) * (1 - config["train"]["lrf"]) + \
    #                config["train"]["lrf"]  # cosine
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    optimizer = torch.optim.AdamW(pg, lr=lr, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["train"]["lr_steps"],
                                                        gamma=config["train"]["lr_gamma"])

    # loss_function = KpLoss()
    loss_function = Kploss_focal(pos_neg_weights=10, gamma=2)

    best_loss = np.inf
    best_ap = 0

    # Resume
    if config["train"]["resume"]:
        if os.path.isfile(config["train"]["resume"]):
            print("=>loading checkpoint {}".format(config["train"]["resume"]))
            ckpt_dict = torch.load(config["train"]["resume"], map_location=device)

            config["train"]["start_epochs"] = ckpt_dict["epoch"]
            # best_acc = ckpt_dict["best_acc"]
            model.load_state_dict(ckpt_dict["state_dict"])
            optimizer.load_state_dict(ckpt_dict['optimizer'])
            lr_scheduler.load_state_dict(ckpt_dict['lr_scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(config["train"]["resume"], ckpt_dict['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(config["train"]["resume"]))

    cudnn.benchmark = True

    # if config["train"]["evaluate"]:
    #     # evaluate
    #     return

    # Train
    for epoch in range(config["train"]["start_epochs"], config["train"]["epochs"]):
        # train one epoch
        pf = '%5s' + '%11s' * 3  # print format
        logging.info(pf % ('Train', 'Epoch', 'Loss', 'Lr'))
        train_loss, train_lr = train_one_epoch(model, train_dataloader, epoch, optimizer, device, RANK, loss_function,
                                               tb_writer, warmup=True, config=config)

        pf = '%5s' + '%11i' * 1 + '%11.6g' * 2  # print format
        logging.info(pf % ('Train', epoch, train_loss, train_lr))

        # 等待所有进程计算完毕
        if RANK != -1:
            if device != torch.device("cpu"):
                torch.cuda.synchronize(device)

        lr_scheduler.step()

        # Test
        if RANK in {-1, 0}:
            pf = '%5s' + '%11s' * 6  # print format
            logging.info(pf % ('Eval', 'Images', 'Precision', 'Recall', 'F1-score', 'AP', 'thresh'))
            test_loss, ap = key_point_eval(model, test_dataloader, epoch, optimizer, device, loss_function, tb_writer,
                                           config)

            # Save model
            is_best_loss = test_loss < best_loss
            is_best = ap > best_ap
            best_loss = min(test_loss, best_loss)
            best_ap = min(ap, best_ap)
            if RANK == 0:
                if is_best:
                    torch.save({
                        'epoch': epoch + 1,
                        'arch': config["train"]["arch"],
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'ap': best_ap
                    }, os.path.join(weights_dir, 'loss_{}_best.pth'.format(config["train"]["arch"])))

                if is_best_loss:
                    torch.save({
                        'epoch': epoch + 1,
                        'arch': config["train"]["arch"],
                        'state_dict': model.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'ap': best_ap
                    }, os.path.join(weights_dir, 'ap_{}_best.pth'.format(config["train"]["arch"])))
            # save_checkpoint(
            #     {
            #         'epoch': epoch + 1,
            #         'arch': config["train"]["arch"],
            #         'state_dict': model.module.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #         'lr_scheduler': lr_scheduler.state_dict(),
            #     },
            #     is_best=is_best,
            #     filename=os.path.join(weights_dir, '{}_{}.pth'.format(config["train"]["arch"], epoch)),
            #     best_filename=os.path.join(weights_dir, '{}_best.pth'.format(config["train"]["arch"]))
            # )

            else:
                if is_best:
                    torch.save({
                        'epoch': epoch + 1,
                        'arch': config["train"]["arch"],
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'ap': best_ap
                    }, os.path.join(weights_dir, '{}_best.pth'.format(config["train"]["arch"])))
                if is_best_loss:
                    torch.save({
                        'epoch': epoch + 1,
                        'arch': config["train"]["arch"],
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'ap': best_ap
                    }, os.path.join(weights_dir, 'ap_{}_best.pth'.format(config["train"]["arch"])))
                    # save_checkpoint(
                    #     {
                    #         'epoch': epoch + 1,
                    #         'arch': config["train"]["arch"],
                    #         'state_dict': model.state_dict(),
                    #         'optimizer': optimizer.state_dict(),
                    #         'lr_scheduler': lr_scheduler.state_dict(),
                    #     },
                    #     is_best=is_best,
                    #     filename=os.path.join(weights_dir, '{}_{}.pth'.format(config["train"]["arch"], epoch)),
                    #     best_filename=os.path.join(weights_dir, '{}_best.pth'.format(config["train"]["arch"]))
                    # )

    if WORLD_SIZE > 1 and RANK == 0:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        logging.info('Destroying process group... ')
        dist.destroy_process_group()


if __name__ == '__main__':
    args = parser_args()
    main(args)
