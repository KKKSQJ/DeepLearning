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
from torchvision import transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset, distributed, dataset, dataloader
import torch.backends.cudnn as cudnn

from utils import increment_path, torch_distributed_zero_first, save_checkpoint, accuracy, AverageMeter, \
    ProgressMeter
from dataLoader import read_split_data, My_Dataset, My_Dataset_with_txt
from models import model_dict, get_model

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
    parser.add_argument('--arch', default='shufflenet_v1_g3', help='model name')
    parser.add_argument('--classes', default=5, help='number of classes')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--freeze_layers', action='store_true')
    parser.add_argument('--syncBN', action='store_true')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--start_epochs', default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", type=str, default='', help="device = 'cpu' or '0' or '0,1,2,3'")
    parser.add_argument('--lr', type=float)
    parser.add_argument("--lrf", type=float)
    parser.add_argument("--print_freq", type=int)
    parser.add_argument('--weights', type=str)
    parser.add_argument('--resume', type=str)

    args = parser.parse_args()

    with open(vars(args)["config_path"], "r", encoding='utf-8') as config_file:
        config = yaml.full_load(config_file)

    if args.data_path is not None:
        config["data"]["data_path"] = args.data_path
    if args.arch is not None:
        config["train"]["arch"] = args.arch
    if args.classes is not None:
        config["train"]["classes"] = args.classes
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

    if RANK in {-1, 0}:
        os.makedirs('data', exist_ok=True)
        data_info = read_split_data(config["data"]["data_path"], save_dir='data', val_rate=0.2, plot_image=True)

    with torch_distributed_zero_first(LOCAL_RANK):
        train_set = My_Dataset_with_txt('data', "train.txt", transform=data_transform["train"])
        test_set = My_Dataset_with_txt('data', "val.txt", transform=data_transform["val"])
        # train_set.data = train_set.data[:100]
        # test_set.data = test_set.data[:100]

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
    nb = len(train_dataloader)
    # warmup_iters = max(round(config["train"]["warmup_epochs"] * nb), 100)

    # Write to tensorboard
    if RANK in {-1, 0}:
        for i in range(10):
            img, target = test_set[i]
            tb_writer.add_image("test_set", img, target)

    # Create model
    assert config["train"]["arch"] in model_dict
    model_build_func = get_model(config["train"]["arch"])
    model = model_build_func(num_classes=config["train"]["classes"])

    # pretrain weights
    mf = False
    if config["train"]["weights"] is not None and os.path.exists(config["train"]["weights"]):
        with torch_distributed_zero_first(LOCAL_RANK):
            checkpoint_path = config["train"]["weights"]

        ckpt = torch.load(checkpoint_path, map_location='cpu')
        ckpt_dict = {k: v for k, v in ckpt.items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(ckpt_dict, strict=True)
    else:
        if RANK != -1:
            checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
            mf = True

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

    if config["train"]["scheduler"].lower() == 'cosine':
        # Scheduler https://arxiv.org/pdf/1812.01187.pdf
        lf = lambda x: ((1 + math.cos(x * math.pi / config["train"]["epochs"])) / 2) * (1 - config["train"]["lrf"]) + \
                       config["train"]["lrf"]  # cosine
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    elif config["train"]["scheduler"].lower() == 'step':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["train"]["lr_steps"],
                                                         gamma=config["train"]["lr_gamma"])

    loss_function = torch.nn.CrossEntropyLoss()

    best_acc = 0
    best_epoch = 0

    # Resume
    if config["train"]["resume"]:
        if os.path.isfile(config["train"]["resume"]):
            print("=>loading checkpoint {}".format(config["train"]["resume"]))
            ckpt_dict = torch.load(config["train"]["resume"], map_location=device)

            config["train"]["start_epochs"] = ckpt_dict["epoch"]
            best_acc = ckpt_dict["best_acc"]
            model.load_state_dict(ckpt_dict["state_dict"])
            optimizer.load_state_dict(ckpt_dict['optimizer'])
            scheduler.load_state_dict(ckpt_dict['scheduler'])
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
        model.train()

        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':6.5f')
        top1 = AverageMeter('Acc@1', ':6.3f')
        top5 = AverageMeter('Acc@5', ':6.3f')
        lr = AverageMeter('Lr', ':6.6f')

        pf = '%5s' + '%11s' * 5  # print format
        logging.info(pf % ('Train', 'epoch', 'top1', 'top5', 'loss', 'lr'))

        if RANK != -1:
            train_dataloader.sampler.set_epoch(epoch)

        progress = ProgressMeter(
            len(train_dataloader),
            [batch_time, data_time, losses, top1, top5, lr],
            prefix="Epoch: [{}]".format(epoch))

        optimizer.zero_grad()
        lr_scheduler = None

        end = time.time()
        for i, (imgs, targets) in enumerate(train_dataloader):
            data_time.update(time.time() - end)
            ni = i + nb * epoch
            imgs = imgs.to(device)
            targets = targets.to(device)

            # Warmup
            # if ni <= warmup_iters:
            #     pass

            # Multi-scale
            # if config["train"]["multi_scale"]:
            #     size = (500, 500)
            #     imgs = nn.functional.interpolate(imgs, size=size, mode='bilinear', align_corners=False)

            # Forward
            # with torch.cuda.amp.autocast(amp):
            pred = model(imgs)
            loss = loss_function(pred, targets)
            loss.backward()
            acc1, acc5 = accuracy(pred, targets, topk=(1, 5))
            losses.update(loss.item(), imgs.size(0))
            top1.update(acc1[0], imgs.size(0))
            top5.update(acc5[0], imgs.size(0))
            lr.update(optimizer.param_groups[0]['lr'])

            optimizer.step()
            optimizer.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()

            if lr_scheduler is not None:
                lr_scheduler.step()

            if RANK in {-1, 0}:
                pf = '%5s' + '%11i' * 1 + '%11.6g' * 4  # print format
                logging.info(pf % ('Train', epoch, top1.avg, top5.avg, losses.avg, lr.avg))

                if i % config["train"]["print_freq"] == 0:
                    progress.display(i)
                tb_writer.add_scalar("train_lr", optimizer.param_groups[0]["lr"], ni)

        # 等待所有进程计算完毕
        if RANK != -1:
            if device != torch.device("cpu"):
                torch.cuda.synchronize(device)

        scheduler.step()

        # Test
        if RANK in {-1, 0}:
            batch_time = AverageMeter('Time', ':6.3f')
            losses = AverageMeter('Loss', ':.4e')
            top1 = AverageMeter('Acc@1', ':6.2f')
            top5 = AverageMeter('Acc@5', ':6.2f')
            progress = ProgressMeter(
                len(test_dataloader),
                [batch_time, losses, top1, top5],
                prefix='Test: ')
            model.eval()

            # sum_num = torch.zeros(1).to(device)
            # pbar = enumerate(test_dataloader)
            # pbar = tqdm(pbar, total=len(test_dataloader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10}')
            pf = '%5s' + '%11s' * 4  # print format
            logging.info(pf % ('Test', 'epoch', 'top1', 'top5', 'lr'))

            with torch.no_grad():
                end = time.time()
                for i, (imgs, targets) in enumerate(test_dataloader):
                    imgs = imgs.to(device)
                    targets = targets.to(device)
                    pred = model(imgs)
                    loss = loss_function(pred, targets)

                    # measure accuracy and record loss
                    acc1, acc5 = accuracy(pred, targets, topk=(1, 5))
                    losses.update(loss.item(), imgs.size(0))
                    top1.update(acc1[0], imgs.size(0))
                    top5.update(acc5[0], imgs.size(0))

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    if i % config["train"]["print_freq"] == 0:
                        progress.display(i)

            tags = ["acc1", "acc5", "learning_rate"]
            tb_writer.add_scalar(tags[0], top1.avg, epoch)
            tb_writer.add_scalar(tags[1], top5.avg, epoch)
            tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)

            pf = '%5s' + '%11i' * 1 + '%11.6g' * 3  # print format
            logging.info(pf % ('Test', epoch, top1.avg, top5.avg, optimizer.param_groups[0]['lr']))

            # Save model
            acc = top1.avg
            is_best = acc > best_acc
            best_acc = max(acc, best_acc)
            if RANK == 0:
                if is_best:
                    best_epoch = epoch
                    # save_checkpoint(
                    #     {
                    #         'epoch': epoch + 1,
                    #         'arch': config["train"]["arch"],
                    #         'state_dict': model.module.state_dict(),
                    #         'best_acc': best_acc,
                    #         'optimizer': optimizer.state_dict(),
                    #         'scheduler': scheduler.state_dict(),
                    #     },
                    #     is_best=is_best,
                    #     filename=os.path.join(weights_dir, '{}_{}.pth'.format(config["train"]["arch"], epoch)),
                    #     best_filename=os.path.join(weights_dir, '{}_best.pth'.format(config["train"]["arch"]))
                    # )
                    torch.save(
                        {
                            'epoch': epoch + 1,
                            'arch': config["train"]["arch"],
                            'state_dict': model.module.state_dict(),
                            'best_acc': best_acc,
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                        },
                        os.path.join(weights_dir, '{}_best.pth'.format(config["train"]["arch"]))
                    )


            else:
                if is_best:
                    best_epoch = epoch
                    # save_checkpoint(
                    #     {
                    #         'epoch': epoch + 1,
                    #         'arch': config["train"]["arch"],
                    #         'state_dict': model.state_dict(),
                    #         'best_acc': best_acc,
                    #         'optimizer': optimizer.state_dict(),
                    #         'scheduler': scheduler.state_dict(),
                    #     },
                    #     is_best=is_best,
                    #     filename=os.path.join(weights_dir, '{}_{}.pth'.format(config["train"]["arch"], epoch)),
                    #     best_filename=os.path.join(weights_dir, '{}_best.pth'.format(config["train"]["arch"]))
                    # )
                    torch.save(
                        {
                            'epoch': epoch + 1,
                            'arch': config["train"]["arch"],
                            'state_dict': model.state_dict(),
                            'best_acc': best_acc,
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                        },
                        os.path.join(weights_dir, '{}_best.pth'.format(config["train"]["arch"]))
                    )

    logging.info(f"best epoch:{best_epoch}, acc:{best_acc}")
    if WORLD_SIZE > 1 and RANK == 0:
        if os.path.exists(checkpoint_path) and mf:
            os.remove(checkpoint_path)

        logging.info('Destroying process group... ')
        dist.destroy_process_group()


if __name__ == '__main__':
    args = parser_args()
    main(args)
