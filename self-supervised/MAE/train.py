import math
import time
from datetime import datetime

import torch
import argparse
import os
import random
import numpy as np

from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast as autocast

from models.MAE import MAEVisonTransformer as MAE
from utils.optimizer_step import Optimizer, build_optimizer
from data.TransformerDataset import ImageDataset
from utils.lr_iter import cosine_learning_rate, step_learning_rate
from loss.mse_loss import MSELoss
from utils.utils import record_log, Metric_rank, record_scalars, translate_state_dict
from utils.LARS import LARC


def parser_arg():
    parser = argparse.ArgumentParser()
    # ------ddp
    # parser.add_argument('--ngpu', type=int, default=1)
    # parser.add_argument('--rank', default=-1, type=int,
    #                     help='node rank for distributed training')
    # parser.add_argument('--dist-backend', default='nccl',
    #                     type=str, help='distributed backend')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--distributed', default=0, type=int,
                        help="use distributed method to training!!")

    # one gpu
    parser.add_argument('--use-gpu', type=bool, default=True)

    # ----- data
    parser.add_argument('--train_file', type=str,
                        default="train.txt")
    parser.add_argument('--val_file', type=str,
                        default="val.txt")
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--color_prob', type=float, default=0.0)

    # ----- checkpoints log dir
    parser.add_argument('--checkpoints-path', default='checkpoints', type=str)
    parser.add_argument('--log-dir', default='logs', type=str)

    # ---- optimizer
    parser.add_argument('--optimizer_name', default="adamw", type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--cosine', default=0, type=int)
    parser.add_argument('--weight_decay', default=5e-2, type=float)

    # --- vit
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--calculate_val', default=0, type=int)
    parser.add_argument('--finetune', default=0, type=int)

    # batchsize
    parser.add_argument('--lars', default=0, type=int, help="use the lars optimizer for big batchsize")
    parser.add_argument('--lars_confience', default=2e-2, type=float)

    # MixUp
    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')

    # ---- train
    parser.add_argument('--model_name', type=str, default='mae')
    parser.add_argument('--warmup_epochs', default=100, type=int)
    parser.add_argument('--max_epochs', default=1000, type=int)

    # ---- eval
    parser.add_argument('--pre-save-epoch', type=int, default=100)

    args = parser.parse_args()
    return args


# random seed
def setup_seed(seed=100):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def train(model, train_loader, optimizer, criterion, epoch, batch_iter, train_batch, total_batch, scaler, log_writer,
          train_metric, mixup_fn, args):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    loader_length = len(train_loader)
    for batch_idx, data in enumerate(train_loader):
        batch_start = time.time()
        # TODO: add the layer-wise lr decay
        if args.cosine:
            # cosine learning rate
            lr = cosine_learning_rate(
                args, epoch, batch_iter, optimizer, train_batch
            )
        else:
            # step learning rate
            lr = step_learning_rate(
                args, epoch, batch_iter, optimizer, train_batch
            )
        # forward
        if args.finetune:
            inputs, targets, path = data[0], data[1], data[2]
            inputs = inputs.to(device)
            targets = targets.to(device)
        else:
            inputs = data
            inputs = inputs.to(device)

        # mixup or cutmix
        if mixup_fn is not None:
            inputs, targets = mixup_fn(inputs, targets)

        with autocast():
            if args.finetune:
                outputs = model(inputs)
                losses = criterion(outputs, targets)
            else:
                pred_mask_pixel_values, mask_patches = model(inputs)
                losses = criterion(pred_mask_pixel_values, mask_patches)

        # translate the miuxp one hot to float
        if mixup_fn is not None:
            targets = targets.argmax(dim=1)

        optimizer.zero_grad()

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        if args.finetune:
            # calculate the accuracy
            batch_acc, _ = accuracy(outputs, targets)

        # record the average momentum result
        train_metric["losses"].update(losses.data.item())
        if args.finetune:
            train_metric["accuracy"].update(batch_acc[0])

        batch_time = time.time() - batch_start

        batch_iter += 1

        if args.local_rank == 0:
            if args.finetune:
                print(
                    "[Training] Time: {} Epoch: [{}/{}] batch_idx: [{}/{}] batch_iter: [{}/{}] batch_losses: {:.4f} batch_accuracy: {:.4f} LearningRate: {:.9f} BatchTime: {:.4f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        epoch,
                        args.max_epochs,
                        batch_idx,
                        train_batch,
                        batch_iter,
                        total_batch,
                        losses.data.item(),
                        batch_acc[0],
                        lr,
                        batch_time
                    ))
            else:
                print(
                    "[Training] Time: {} Epoch: [{}/{}] batch_idx: [{}/{}] batch_iter: [{}/{}] batch_losses: {:.4f} LearningRate: {:.9f} BatchTime: {:.4f}".format(
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        epoch,
                        args.max_epochs,
                        batch_idx,
                        train_batch,
                        batch_iter,
                        total_batch,
                        losses.data.item(),
                        lr,
                        batch_time
                    ))

        if args.local_rank == 0:
            # batch record
            if args.finetune:
                record_log(log_writer, losses, lr, batch_iter, batch_time, batch_acc[0])
            else:
                record_log(log_writer, losses, lr, batch_iter, batch_time, None)

    if args.local_rank == 0:
        # epoch record
        if args.finetune:
            record_scalars(log_writer, train_metric["losses"].average, epoch, flag="train",
                           mean_acc=train_metric["accuracy"].average)
        else:
            record_scalars(log_writer, train_metric["losses"].average, epoch, flag="train", mean_acc=None)

    return batch_iter, scaler, train_metric['losses'].average


def eval(model, val_loader, epoch, log_writer):
    """Validation and get the metric
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_losses, epoch_accuracy = 0.0, 0.0
    criterion = nn.CrossEntropyLoss()

    batch_acc_list = []
    batch_loss_list = []

    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            inputs, targets, _ = data[0], data[1], data[2]

            inputs = inputs.to(device)
            targets = targets.to(device)

            batch_size = inputs.shape[0]

            with autocast():
                outputs = model(inputs)
                losses = criterion(outputs, targets)

            batch_accuracy, _ = accuracy(outputs, targets)

            batch_acc_list.append(batch_accuracy[0])
            batch_loss_list.append(losses.data.item())

    epoch_acc = np.mean(batch_acc_list)
    epoch_loss = np.mean(batch_loss_list)

    # all reduce the correct number
    # dist.all_reduce(epoch_accuracy, op=dist.ReduceOp.SUM)

    if args.local_rank == 0:
        print(
            f"Validation Epoch: [{epoch}/{args.max_epochs}] Epoch_mean_losses: {epoch_loss} Epoch_mean_accuracy: {epoch_acc}")

        record_scalars(log_writer, epoch_loss, epoch, flag="val", mean_acc=epoch_acc)

    return epoch_loss, epoch_acc


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        crr = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            acc = correct_k.mul_(1 / batch_size).item()
            res.append(acc)  # unit: percentage (%)
            crr.append(correct_k)
        return res, crr


def main(args):
    total_rank = torch.cuda.device_count()
    print('rank: {} / {}'.format(args.local_rank, total_rank))
    # dist.init_process_group(backend=args.dist_backend)
    torch.cuda.set_device(0)

    ngpus_per_node = total_rank

    if args.local_rank == 0:
        if not os.path.exists(args.checkpoints_path):
            os.makedirs(args.checkpoints_path)

    # metric
    train_losses_metric = Metric_rank("train_losses")
    train_accuracy_metric = Metric_rank("train_accuracy")
    train_metric = {
        "losses": train_losses_metric,
        'accuracy': train_accuracy_metric
    }

    # model MAE vit

    if args.finetune:
        model = MAE(
            image_size=224,
            patch_size=16,
            encoer_dim=512,
            mlp_dim=1024,
            encoder_depth=6,
            num_encoder_head=8,
            dim_per_head=64,
            decoder_dim=512,
            decoder_depth=6,
            num_decoder_head=8,
            mask_ratio=0.75
        )
    else:
        # vit tiny & vit base
        # vit base decoder: 512 8 16
        model = MAE(
            image_size=args.crop_size,
            patch_size=args.patch_size,
            encoer_dim=768,
            mlp_dim=1024,
            encoder_depth=12,
            num_encoder_head=12,
            dim_per_head=64,
            decoder_dim=512,
            decoder_depth=8,
            num_decoder_head=16,
            mask_ratio=0.75
        )

    if args.local_rank == 0:
        print(f"===============model arch ===============")
        print(model)

    # model mode
    model.train()

    if torch.cuda.is_available():
        model.cuda()

    # optimizer
    print("optimizer name: ", args.optimizer_name)
    optimizer = Optimizer(args.optimizer_name)(
        param=model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        finetune=args.finetune
    )

    if args.lars:
        optimizer = LARC(optimizer, trust_coefficient=args.lars_confience)

    # print(optimizer)
    if args.distributed:
        model = DataParallel(model,
                             device_ids=[args.local_rank],
                             find_unused_parameters=True)

    # dataset & dataloader
    train_dataset = ImageDataset(
        image_file=args.train_file,
        train_phase=True,
        crop_size=args.crop_size,
        shuffle=True,
        interpolation="bilinear",
        auto_augment="rand",
        color_prob=args.color_prob,
        hflip_prob=0.5
    )

    validation_dataset = ImageDataset(
        image_file=args.val_file,
        train_phase=False,
        crop_size=args.crop_size,
        shuffle=False
    )

    if args.local_rank == 0:
        print("Trainig dataset length: ", len(train_dataset))
        print("Validation dataset length: ", len(validation_dataset))

    # sampler
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        validation_sampler = DistributedSampler(validation_dataset)
    else:
        train_sampler = None
        validation_sampler = None

    # logs
    log_writer = SummaryWriter(args.log_dir)

    # dataloader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=False,
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=args.batch_size,
        shuffle=(validation_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=validation_sampler,
        drop_last=False
    )

    # mixup & cutmix
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active and args.finetune:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        print("use the mixup function ")

    start_epoch = 1
    batch_iter = 0
    train_batch = math.ceil(len(train_dataset) / (args.batch_size * ngpus_per_node))
    total_batch = train_batch * args.max_epochs
    no_warmup_total_batch = int(args.max_epochs - args.warmup_epochs) * train_batch

    scaler = torch.cuda.amp.GradScaler()

    if args.finetune:
        # loss
        if args.mixup > 0.:
            # smoothing is handled with mixup label transform
            criterion = SoftTargetCrossEntropy()
        elif args.smoothing:
            criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = MSELoss()

    best_loss, best_acc = np.inf, 0.0
    # training loop
    for epoch in range(start_epoch, args.max_epochs + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        batch_iter, scalerm, loss = train(model, train_loader, optimizer, criterion, epoch, batch_iter, train_batch,
                                          total_batch, scaler, log_writer, train_metric, mixup_fn, args)

        if epoch % args.pre_save_epoch == 0:
            if args.calculate_val:
                # calculate the validation with the batch iter
                val_loss, val_acc = eval(model, validation_loader, epoch, log_writer)
                # recored & write
                if args.local_rank == 0:
                    best_loss = val_loss
                    state_dict = translate_state_dict(model.state_dict())
                    state_dict = {
                        'epoch': epoch,
                        'state_dict': state_dict,
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(
                        state_dict,
                        args.checkpoints_path + '/' 'r50' + f'_losses_{best_loss}' + '.pth'
                    )

                    best_acc = val_acc
                    state_dict = translate_state_dict(model.state_dict())
                    state_dict = {
                        'epoch': epoch,
                        'state_dict': state_dict,
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(state_dict,
                               args.checkpoints_path + '/' + 'vit_finetune' + f'_accuracy_{best_acc}' + '.pth')
            else:
                if args.local_rank == 0:
                    state_dict = translate_state_dict(model.state_dict())
                    state_dict = {
                        'epoch': epoch,
                        'state_dict': state_dict,
                        'optimizer': optimizer.state_dict(),
                    }
                    torch.save(
                        state_dict,
                        args.checkpoints_path + '/' 'vit-mae' + f'_losses_{train_metric["losses"].average}' + '.pth'
                    )
                    if loss < best_loss:
                        best_loss = loss
                        torch.save(
                            state_dict,
                            args.checkpoints_path + '/' 'best-vit-mae.pth'
                        )
        # model mode
        model.train()


if __name__ == '__main__':
    args = parser_arg()
    setup_seed()
    main(args)
    print(args)
