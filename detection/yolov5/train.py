# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset

Usage:
    $ python path/to/train.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, SGD, lr_scheduler
from tqdm import tqdm

# 获取脚本的绝对路径
FILE = Path(__file__).absolute()
# 将该脚本的父路径添加到系统路径中，方便脚本找到对应模快
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

import val  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr, methods
from utils.downloads import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_labels, plot_evolve
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, de_parallel
from utils.loggers.wandb.wandb_utils import check_wandb_resume
from utils.metrics import fitness
from utils.loggers import Loggers
from utils.callbacks import Callbacks

# 实例化日志
LOGGER = logging.getLogger(__name__)
# os.getenv():获取环境变量
# DDP参数
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html  # 每台机子上的进程的序号
RANK = int(os.getenv('RANK', -1))               # 当前进行的序号，用于进程间的通讯，0：为master进程
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))    # 全局并行数量

"""
hpy:'data/hyps/hyp.scratch.yaml'
opt:Namespace(命令行参数)
device:训练的设备:cuda:0
callbacks:回调函数，处理 YOLOv5 Hooks 的所有注册回调
"""
def train(hyp,  # path/to/hyp.yaml or hyp dictionary
          opt,
          device,
          callbacks=Callbacks()
          ):
    # 从命令行参数中获取对应参数
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, = \
        Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, opt.data, opt.cfg, \
        opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze

    # Directories
    # 创建保存权重的文件夹路径
    w = save_dir / 'weights'  # weights dir
    w.mkdir(parents=True, exist_ok=True)  # make dir
    # 权重文件的名称
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        # 读取模型训练超参数
        with open(hyp) as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # Save run settings
    # 将本次训练的超参数保存到本地目录
    # yaml文件，读取：yaml.safe_load(stream)。保存：yaml.safe_dump(data,stream)
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)  # vars():转换为dict
    data_dict = None

    # Loggers
    if RANK in [-1, 0]:
        # 初始化logger实例，用于记录训练日志
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance
        if loggers.wandb:
            data_dict = loggers.wandb.data_dict
            # resume:是否接着上次的训练结果接着训练
            if resume:
                weights, epochs, hyp = opt.weights, opt.epochs, opt.hyp

        # Register actions
        # 将loggers中的内容放入注册表中
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

    # Config
    # 画图
    plots = not evolve  # create plots
    cuda = device.type != 'cpu'
    # 初始化随机种子，
    init_seeds(1 + RANK)
    # 分布式训练中所有进程等待主进程加载数据
    with torch_distributed_zero_first(RANK):
        # check_dataset(data):检查本地是否存在对应数据集，如果不存在则下载并且解压在本地
        data_dict = data_dict or check_dataset(data)  # check if None
    # 训练集路径，验证集路径
    train_path, val_path = data_dict['train'], data_dict['val']
    # 类别数量
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    # 类别对应的名字
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check
    is_coco = data.endswith('coco.yaml') and nc == 80  # COCO dataset

    # Model
    pretrained = weights.endswith('.pt')
    if pretrained:
        # 等待主线程加载预训练权重，如果本地没有，则下载
        with torch_distributed_zero_first(RANK):
            weights = attempt_download(weights)  # download if not found locally
        # 加载模型权重
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        # 创建网络结构，Model在yolo.py文件中，ch=3为输入通道。RGB
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        # 如果opt.cfg存在(表示采用预训练权重进行训练)就设置去除anchor
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        # 显示加载预训练权重的的键值对和创建模型的键值对
        # 如果设置了resume，则会少加载两个键值对(anchors,anchor_grid)
        LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

    # Freeze
    """
    冻结模型层,设置冻结层名字即可
    具体可以查看https://github.com/ultralytics/yolov5/issues/679
    但作者不鼓励冻结层,因为他的实验当中显示冻结层不能获得更好的性能,参照:https://github.com/ultralytics/yolov5/pull/707
    并且作者为了使得优化参数分组可以正常进行,在下面将所有参数的requires_grad设为了True
    其实这里只是给一个freeze的示例
    """
    freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False

    # Optimizer
    """
    nbs为模拟的batch_size; 
    就比如默认的话上面设置的opt.batch_size为16,这个nbs就为64，
    也就是模型梯度累积了64/16=4(accumulate)次之后
    再更新一次模型，变相的扩大了batch_size
    """
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    # 根据accumulate设置权重衰减系数
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    LOGGER.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    # 将模型分成三组(weight、bn, bias, 其他所有参数)优化
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    # 选用优化器，并设置pg0组的优化方式
    if opt.adam:
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    # 设置weight、bn的优化方式
    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    # 设置biases的优化方式
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    # 打印优化信息
    LOGGER.info(f"{colorstr('optimizer:')} {type(optimizer).__name__} with parameter groups "
                f"{len(g0)} weight, {len(g1)} weight (no decay), {len(g2)} bias")
    del g0, g1, g2

    # Scheduler
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    # 是否使用线性学习率衰减，默认还是使用的余弦退火衰减
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        # 设置学习率衰减，这里为余弦退火方式进行衰减
        # 就是根据one_cycle中定义的公式，lf,epoch和超参数hyp['lrf']进行衰减
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    # 可视化scheduler
    # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    # 为模型创建EMA指数滑动平均,如果GPU进程数大于1,则不创建
    ema = ModelEMA(model) if RANK in [-1, 0] else None

    # Resume
    # 初始化开始训练的epoch和最好的结果
    # best_fitness是以[0.0, 0.0, 0.1, 0.9]为系数并乘以[精确度, 召回率, mAP@0.5, mAP@0.5:0.95]再求和所得
    # 根据best_fitness来保存best.pt
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        # 加载优化器与best_fitness
        if ckpt['optimizer'] is not None:
            optimizer.load_state_dict(ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        # 加载ema模型和updates参数,保持ema的平滑性,现在yolov5是ema和model都保存了，
        # 确实这样在resume的时候更合理一些，只是训练时保存的pt文件会更大
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Epochs
        # 加载训练的轮次
        start_epoch = ckpt['epoch'] + 1
        if resume:
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
        """
        如果新设置epochs小于加载的epoch，
        则视新设置的epochs为需要再训练的轮次数而不再是总的轮次数
        """
        if epochs < start_epoch:
            LOGGER.info(f"{weights} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {epochs} more epochs.")
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, csd

    # Image sizes
    # 获取模型最大步长
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    # 获取模型FPN层数
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    # 检查输入图片分辨率确保能够整除总步长gs
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # DP mode
    # 分布式训练,参照:https://github.com/ultralytics/yolov5/issues/475
    # DataParallel模式,仅支持单机多卡
    # rank为进程编号, 这里应该设置为rank=-1则使用DataParallel模式
    # rank=-1且gpu数量=1时,不会进行分布式
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        logging.warning('DP not recommended, instead use torch.distributed.run for best DDP Multi-GPU results.\n'
                        'See Multi-GPU Tutorial at https://github.com/ultralytics/yolov5/issues/475 to get started.')
        model = torch.nn.DataParallel(model)

    # SyncBatchNorm
    # 使用跨卡同步BN
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')

    # Trainloader
    # 创建训练集dataloader
    train_loader, dataset = create_dataloader(train_path, imgsz, batch_size // WORLD_SIZE, gs, single_cls,
                                              hyp=hyp, augment=True, cache=opt.cache, rect=opt.rect, rank=RANK,
                                              workers=workers, image_weights=opt.image_weights, quad=opt.quad,
                                              prefix=colorstr('train: '))
    mlc = int(np.concatenate(dataset.labels, 0)[:, 0].max())  # max label class
    nb = len(train_loader)  # number of batches
    assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'

    # Process 0
    if RANK in [-1, 0]:
        val_loader = create_dataloader(val_path, imgsz, batch_size // WORLD_SIZE * 2, gs, single_cls,
                                       hyp=hyp, cache=None if noval else opt.cache, rect=True, rank=-1,
                                       workers=workers, pad=0.5,
                                       prefix=colorstr('val: '))[0]

        if not resume:
            labels = np.concatenate(dataset.labels, 0)
            # c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                plot_labels(labels, names, save_dir)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision

        callbacks.on_pretrain_routine_end()

    # DDP mode
    if cuda and RANK != -1:
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= nc / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names

    # Start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations, max(3 epochs, 1k iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    # 自动混合精度训练
    scaler = amp.GradScaler(enabled=cuda)
    # 初始化计算loss的类
    compute_loss = ComputeLoss(model)  # init loss class
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers} dataloader workers\n'
                f'Logging results to {save_dir}\n'
                f'Starting training for {epochs} epochs...')
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)
        if opt.image_weights:
            # Generate indices
            if RANK in [-1, 0]:
                cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights
                iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights
                dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx
            # Broadcast if DDP
            if RANK != -1:
                indices = (torch.tensor(dataset.indices) if RANK == 0 else torch.zeros(dataset.n)).int()
                dist.broadcast(indices, 0)
                if RANK != 0:
                    dataset.indices = indices.cpu().numpy()

        # Update mosaic border
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(3, device=device)  # mean losses
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)
        LOGGER.info(('\n' + '%10s' * 7) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'labels', 'img_size'))
        if RANK in [-1, 0]:
            pbar = tqdm(pbar, total=nb)  # progress bar
        optimizer.zero_grad()
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device, non_blocking=True).float() / 255.0  # uint8 to float32, 0-255 to 0.0-1.0

            # Warmup
            if ni <= nw:
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            if opt.multi_scale:
                sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            # 混合半精度训练
            with amp.autocast(enabled=cuda):
                pred = model(imgs)  # forward
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:
                    loss *= 4.

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)
                last_opt_step = ni

            # Log
            if RANK in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                pbar.set_description(('%10s' * 2 + '%10.4g' * 5) % (
                    f'{epoch}/{epochs - 1}', mem, *mloss, targets.shape[0], imgs.shape[-1]))
                callbacks.on_train_batch_end(ni, model, imgs, targets, paths, plots)
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x['lr'] for x in optimizer.param_groups]  # for loggers
        scheduler.step()

        if RANK in [-1, 0]:
            # mAP
            callbacks.on_train_epoch_end(epoch=epoch)
            ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
            final_epoch = epoch + 1 == epochs
            if not noval or final_epoch:  # Calculate mAP
                results, maps, _ = val.run(data_dict,
                                           batch_size=batch_size // WORLD_SIZE * 2,
                                           imgsz=imgsz,
                                           model=ema.ema,
                                           single_cls=single_cls,
                                           dataloader=val_loader,
                                           save_dir=save_dir,
                                           save_json=is_coco and final_epoch,
                                           verbose=nc < 50 and final_epoch,
                                           plots=plots and final_epoch,
                                           callbacks=callbacks,
                                           compute_loss=compute_loss)

            # Update best mAP
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.on_fit_epoch_end(log_vals, epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': deepcopy(de_parallel(model)).half(),
                        'ema': deepcopy(ema.ema).half(),
                        'updates': ema.updates,
                        'optimizer': optimizer.state_dict(),
                        'wandb_id': loggers.wandb.wandb_run.id if loggers.wandb else None}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                del ckpt
                callbacks.on_model_save(last, epoch, final_epoch, best_fitness, fi)

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in [-1, 0]:
        LOGGER.info(f'\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.')
        if not evolve:
            if is_coco:  # COCO dataset
                for m in [last, best] if best.exists() else [last]:  # speed, mAP tests
                    results, _, _ = val.run(data_dict,
                                            batch_size=batch_size // WORLD_SIZE * 2,
                                            imgsz=imgsz,
                                            model=attempt_load(m, device).half(),
                                            iou_thres=0.7,  # NMS IoU threshold for best pycocotools results
                                            single_cls=single_cls,
                                            dataloader=val_loader,
                                            save_dir=save_dir,
                                            save_json=True,
                                            plots=False)
            # Strip optimizers
            for f in last, best:
                if f.exists():
                    strip_optimizer(f)  # strip optimizers
        callbacks.on_train_end(last, best, plots, epoch)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")

    torch.cuda.empty_cache()
    return results


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # 预训练权重路径
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    # 模型配置文件路径
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    # 训练数据配置文件路径
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    # 训练超参数配置文件路径
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch.yaml', help='hyperparameters path')
    # 训练的epochs
    parser.add_argument('--epochs', type=int, default=300)
    # batch-size，不同大小的模型使用不同的bs
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    # 训练和验证时输入网络的图片大小
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    # 是否采用矩阵训练，默认False，action='':表示触发是为true
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    # 接着上次的训练结果继续训练
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    # 只保存最后一个模型，默认False
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    # 只验证最后一个epoch，默认False
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    # 不自动调整anchor，默认False
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    # 是否进行超参数进化，默认False
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    # 谷歌云盘，一般不会用到
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    # 提前缓存图片到内存(或者硬盘)，以加快训练速度
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    # 训练时是否对图像进行采样的权重
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    # 训练的设备
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 是否多尺度训练，默认False
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    # 是否将多类别数据当做一个类训练，默认False
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    # 是否使用adam优化器
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    # 是否使用跨卡同步BN，只在DDP模型使用
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    # dataloader的最大worker数量
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    # 训练日志，权重保存的目录
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    # wandb相关参数
    parser.add_argument('--entity', default=None, help='W&B entity')
    # 数据集名字，默认exp
    parser.add_argument('--name', default='exp', help='save to project/name')
    # 如果存在保存的目录，则不重新创建该目录
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # 是否使用collate_fn4作为dataloader的选择函数
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    # 是否使用线性学习率，默认不使用
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    # 是否使用标签平滑
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    # wandb相关参数，上传数据
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    # wandb中log边框的间隔
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    # wandb保存模型的间隔
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    # wandb相关参数
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    # ddp参数
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    # 是否冻结权重
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def main(opt):
    # Checks
    # 初始化logging
    set_logging(RANK)
    # RAND=-1
    if RANK in [-1, 0]:
        # 打印参数
        print(colorstr('train: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
        # 检查当前代码是否为最新版本，如果代码过时，执行git pull
        check_git_status()
        # 检查项目/requirements.txt的环境
        # FILE.parent:当前文件的父目录，即path/yolov5,类型为：windowsPath
        check_requirements(requirements=FILE.parent / 'requirements.txt', exclude=['thop'])

    # Resume
    # 恢复中断的运行，默认为False,不执行该代码块
    if opt.resume and not check_wandb_resume(opt) and not opt.evolve:  # resume an interrupted run
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()  # specified or most recent path
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
        LOGGER.info(f'Resuming training from {ckpt}')
    else:
        # 检查配置文件是否存在，返回文件路径
        opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
        # assert:断言语句。必须指定--cfg或者weights
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve: # 默认为None,即不进行超参数进化
            opt.project = 'runs/evolve'
            opt.exist_ok = opt.resume
        # 日志，权重的保存路径，如runs/train/exp
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    # 选择训练的设备，是使用多卡训练还是单gpu训练还是cpu训练
    device = select_device(opt.device, batch_size=opt.batch_size)
    # 如果使用DDP训练，则初始化DDP环境
    if LOCAL_RANK != -1:
        from datetime import timedelta
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        assert opt.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        assert not opt.image_weights, '--image-weights argument is not compatible with DDP training'
        assert not opt.evolve, '--evolve argument is not compatible with DDP training'
        assert not opt.sync_bn, '--sync-bn known training issue, see https://github.com/ultralytics/yolov5/issues/3998'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    # opt.evolve:None.开始训练
    if not opt.evolve:
        train(opt.hyp, opt, device)
        if WORLD_SIZE > 1 and RANK == 0:
            _ = [print('Destroying process group... ', end=''), dist.destroy_process_group(), print('Done.')]

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
                'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
                'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
                'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
                'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
                'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
                'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
                'box': (1, 0.02, 0.2),  # box loss gain
                'cls': (1, 0.2, 4.0),  # cls loss gain
                'cls_pw': (1, 0.5, 2.0),  # cls BCELoss positive_weight
                'obj': (1, 0.2, 4.0),  # obj loss gain (scale with pixels)
                'obj_pw': (1, 0.5, 2.0),  # obj BCELoss positive_weight
                'iou_t': (0, 0.1, 0.7),  # IoU training threshold
                'anchor_t': (1, 2.0, 8.0),  # anchor-multiple threshold
                'anchors': (2, 2.0, 10.0),  # anchors per output grid (0 to ignore)
                'fl_gamma': (0, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
                'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
                'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
                'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
                'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
                'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
                'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
                'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
                'perspective': (0, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
                'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
                'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
                'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
                'mixup': (1, 0.0, 1.0),  # image mixup (probability)
                'copy_paste': (1, 0.0, 1.0)}  # segment copy-paste (probability)

        with open(opt.hyp) as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if 'anchors' not in hyp:  # anchors commented in hyp.yaml
                hyp['anchors'] = 3
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / 'hyp_evolve.yaml', save_dir / 'evolve.csv'
        if opt.bucket:
            os.system(f'gsutil cp gs://{opt.bucket}/evolve.csv {save_dir}')  # download evolve.csv if exists

        for _ in range(opt.evolve):  # generations to evolve
            if evolve_csv.exists():  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 7] * v[i])  # mutate

            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            results = train(hyp.copy(), opt, device)

            # Write mutation results
            print_mutation(results, hyp.copy(), save_dir, opt.bucket)

        # Plot results
        plot_evolve(evolve_csv)
        print(f'Hyperparameter evolution finished\n'
              f"Results saved to {colorstr('bold', save_dir)}\n"
              f'Use best hyperparameters example: $ python train.py --hyp {evolve_yaml}')


def run(**kwargs):
    # Usage: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
