import os
import os.path
import sys
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
from torchvision import transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset, distributed, dataset, dataloader

from utils import increment_path, torch_distributed_zero_first, reduce_value

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', -1))
print("LOCAL RANK:", LOCAL_RANK)
print("RANK:", RANK)
print("WORLD_SIZE:", WORLD_SIZE)


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='config/train.yaml')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--data_path', type=str, default='data/')
    parser.add_argument('--freeze_layers', action='store_true')
    parser.add_argument('--syncBN', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--start_epochs', default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", type=str, default='', help="device = 'cpu' or '0' or '0,1,2,3'")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument('--weights', type=str, default='resnet50-19c8e357.pth')

    args = parser.parse_args()

    with open(vars(args)["config_path"], "r", encoding='utf-8') as config_file:
        config = yaml.full_load(config_file)

    if args.data_path is not None:
        config["data"]["data_path"] = args.data_path
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
    data_transform = {
        "train": transforms.Compose(
            [transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        ),
        "val": transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
    }

    with torch_distributed_zero_first(LOCAL_RANK):
        train_set = torchvision.datasets.CIFAR100(root=config["data"]["data_path"], train=True,
                                                  transform=data_transform["train"], download=True)
        test_set = torchvision.datasets.CIFAR100(root=config["data"]["data_path"], train=False,
                                                 transform=data_transform["val"], download=True)
        # train_set.data = train_set.data[:1000]
        # test_set.data = test_set.data[:1000]

    nclass = len(train_set.classes)
    # print("nclass:", nclass)
    train_sampler = None if LOCAL_RANK == -1 else distributed.DistributedSampler(train_set, shuffle=True)
    test_sampler = None if LOCAL_RANK == -1 else distributed.DistributedSampler(test_set, shuffle=False)

    nw = min([os.cpu_count(), config["train"]["batch_size"] if config["train"]["batch_size"] > 1 else 0, 8])
    train_dataloader = DataLoader(train_set, batch_size=config["train"]["batch_size"], shuffle=train_sampler is None,
                                  num_workers=nw, sampler=train_sampler, pin_memory=True, drop_last=True)

    test_dataloader = DataLoader(test_set, batch_size=config["train"]["batch_size"], shuffle=False,
                                 num_workers=nw, sampler=test_sampler, pin_memory=True, drop_last=False)
    nb = len(train_dataloader)
    nwarmup = max(round(config["train"]["warmup_epochs"] * nb), 100)

    # Write to tensorboard
    if RANK in {-1, 0}:
        for i in range(10):
            img, target = test_set[i]
            tb_writer.add_image("test_set", img, target)

    # Create model
    checkpoint_path = ""
    model = resnet50(pretrained=False, num_classes=nclass)
    # pretrain weights
    if config["train"]["weights"] is not None and os.path.exists(config["train"]["weights"]):
        with torch_distributed_zero_first(LOCAL_RANK):
            checkpoint_path = config["train"]["weights"]

        ckpt = torch.load(checkpoint_path, map_location='cpu')
        ckpt_dict = {k: v for k, v in ckpt.items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(ckpt_dict, strict=False)
    else:
        checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")

        if RANK == 0:
            torch.save(model.state_dict(), checkpoint_path)
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
    optimizer = torch.optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5e-4)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / config["train"]["epochs"])) / 2) * (1 - config["train"]["lrf"]) + \
                   config["train"]["lrf"]  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    loss_function = torch.nn.CrossEntropyLoss()

    best_acc = 0
    # Train
    for epoch in range(config["train"]["start_epochs"], config["train"]["epochs"]):
        model.train()

        if RANK != -1:
            # train_sampler.set_epoch(epoch)
            train_dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(train_dataloader)
        logging.info(('%20s' * 3) % ('Train Epoch', 'gpu_mem', 'loss'))

        if RANK in {-1, 0}:
            pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10}')

        optimizer.zero_grad()
        mean_loss = torch.zeros(1).to(device)

        for i, (imgs, targets) in pbar:
            ni = i + nb * epoch
            imgs = imgs.to(device)
            targets = targets.to(device)

            # Warmup
            # if ni <= nwarmup:
            #     xi = [0, nwarmup]
            #     for j, x in enumerate(optimizer.param_groups):
            # x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
            # if 'momentum' in x:
            #     x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            # if config["train"]["multi_scale"]:
            #     size = (500, 500)
            #     imgs = nn.functional.interpolate(imgs, size=size, mode='bilinear', align_corners=False)

            # Forward
            # with torch.cuda.amp.autocast(amp):
            pred = model(imgs)
            loss = loss_function(pred, targets)
            loss.backward()
            # if RANK != -1:
            #     loss = reduce_value(loss, WORLD_SIZE, average=True)
            mean_loss = (mean_loss * i + loss.detach()) / (i + 1)

            optimizer.step()
            optimizer.zero_grad()

            # Log
            if RANK in {-1, 0}:
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('epoch: %2s\t\t' + 'mem: %2s\t\t' + 'loss: %2.4g\t\t' + 'lr: %2f') %
                                     (f'{epoch}/{config["train"]["epochs"] - 1}', mem, mean_loss.item(),
                                      optimizer.param_groups[0]['lr']))

        # 等待所有进程计算完毕
        if RANK != -1:
            if device != torch.device("cpu"):
                torch.cuda.synchronize(device)

        scheduler.step()

        # Test
        if RANK in {-1, 0}:
            logging.info(('%20s' * 3) % (f'{epoch}', f'{mem}', f'{mean_loss.item():.3}'))
            model.eval()
            sum_num = torch.zeros(1).to(device)
            pbar = enumerate(test_dataloader)
            # logging.info(('%10s' * 4) % ('Test', 'Epoch', 'gpu_mem', 'loss'))
            pbar = tqdm(pbar, total=len(test_dataloader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10}')

            with torch.no_grad():
                for i, (imgs, targets) in pbar:
                    imgs = imgs.to(device)
                    targets = targets.to(device)
                    pred = model(imgs)
                    pred = torch.max(pred, dim=1)[1]
                    sum_num += torch.eq(pred, targets).sum()

            acc = sum_num / len(test_dataloader.sampler)

            tags = ["loss", "accuracy", "learning_rate"]
            tb_writer.add_scalar(tags[0], mean_loss, epoch)
            tb_writer.add_scalar(tags[1], acc, epoch)
            tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
            logging.info(('%20s' * 4) % ('Test Epoch', 'Loss', 'Acc', 'Lr'))
            logging.info(('%20s' * 4) % (
                f'{epoch}', f'{mean_loss.item():.3}', f'{acc.item():.3}', f'{optimizer.param_groups[0]["lr"]:.3}'))

            # Save model
            if RANK != -1:
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.module.state_dict(), os.path.join(weights_dir, "best.pth"))
                torch.save(model.module.state_dict(), os.path.join(weights_dir, f"model_{epoch}.pth"))
            else:
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), os.path.join(weights_dir, "best.pth"))
                torch.save(model.state_dict(), os.path.join(weights_dir, f"model_{epoch}.pth"))

    if WORLD_SIZE > 1 and RANK == 0:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)

        logging.info('Destroying process group... ')
        dist.destroy_process_group()


if __name__ == '__main__':
    args = parser_args()
    main(args)
