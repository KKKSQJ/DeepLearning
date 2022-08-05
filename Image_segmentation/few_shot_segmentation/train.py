from copy import deepcopy
import yaml
import os
from tqdm import tqdm
import sys
from pathlib import Path
import argparse
import logging
import time
import warnings

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import FewShot
from models import SSPNet
from utils import select_device, count_params, set_seed, mIOU, increment_path

warnings.filterwarnings("ignore")

# 获取脚本的绝对路径
FILE = Path(__file__).absolute()
# 将该脚本的父路径添加到系统路径中，方便脚本找到对应模快
sys.path.append(FILE.parents[0].as_posix())


def parser_args():
    parser = argparse.ArgumentParser(description="Few-shot Segmentation")
    parser.add_argument("--config-path", type=str, help='path of config file', default='config/train.yaml')
    # 其他参数去example.yaml中进行配置，下面参数是为了方便在服务器上进行跑
    parser.add_argument("--batch-size", type=int, help="batch-size")
    parser.add_argument("--device", type=str, help="device = 'cpu' or '0' or '0,1,2,3'")
    parser.add_argument("--shot", type=int, help="k shot")
    parser.add_argument("--size",type=int, help='--input-size')
    parser.add_argument("--fold",type=int, help='n way fold')
    # parser.add_argument('--project', default='runs/train', help='save to project/name')
    # parser.add_argument('--name', default='exp', help='save to project/name')
    # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser_args = parser.parse_args()

    with open(vars(parser_args)["config_path"], "r", encoding='utf-8') as config_file:
        config = yaml.full_load(config_file)

    if parser_args.device is not None:
        config["train"]["device"] = parser_args.device
    if parser_args.batch_size is not None:
        config["train"]["batch_size"] = parser_args.batch_size
    if parser_args.shot is not None:
        config["train"]["shot"] = parser_args.shot
    if parser_args.size is not None:
        config["train"]["input_size"] = parser_args.size
    if parser_args.fold is not None:
        config["train"]["fold"] = parser_args.fold

    return config


def evaluate(model, dataloader, device, config):
    tbar = tqdm(dataloader)
    num_classes = config["train"]["num_class"] + 1
    metric = mIOU(num_classes)

    for i, (img_s_list, mask_s_list, img_q, mask_q, cls, _, id_q) in enumerate(tbar):
        img_q, mask_q = img_q.to(device), mask_q.to(device)
        for k in range(len(img_s_list)):
            img_s_list[k], mask_s_list[k] = img_s_list[k].to(device), mask_s_list[k].to(device)
        cls = cls[0].item()

        with torch.no_grad():
            out_ls = model(img_s_list, mask_s_list, img_q, mask_q)
            pred = torch.argmax(out_ls[0], dim=1)

        pred[pred == 1] = cls
        mask_q[mask_q == 1] = cls

        metric.add_batch(pred.cpu().numpy(), mask_q.cpu().numpy())

        tbar.set_description("Testing mIOU: %.2f" % (metric.evaluate() * 100.0))
    logging.info("Testing mIOU: %.2f" % (metric.evaluate() * 100.0))

    return metric.evaluate() * 100.0


def main(config):
    print(config)

    # Create logging
    project = config["train"]["project"]
    name = config["train"]["name"]
    exist_ok = config["train"]["exist_ok"]
    save_dir = Path(str(increment_path(Path(project) / name, exist_ok=exist_ok)))
    save_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    tb_writer = SummaryWriter(str(save_dir))

    with open(save_dir / 'train.yaml', 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # logging_name = config["train"]["logging_name"]
    # if logging_name is None:
    #     logging_name = f"model_{config['train']['model_name']}_dataset_{config['data']['data_name']}_k_{config['train']['shot']}_fold_{config['train']['fold']}"

    # weights_dir = os.path.join(save_dir, logging_name, "weights")
    # log_dir = os.path.join(save_dir, logging_name, "logs")
    # os.makedirs(weights_dir, exist_ok=True)
    # os.makedirs(log_dir, exist_ok=True)
    # logging_path = os.path.join(log_dir, "train.log")

    logging_path = str(save_dir / "train.log")
    logging.basicConfig(filename=logging_path, level=logging.INFO, filemode="w+")
    for k, v in config.items():
        logging.info(f"====>  {k}: {v}   <=====")

    # Create Data
    img_path = config["data"]["img_path"]
    mask_path = config["data"]["mask_path"]
    crop_size = config["train"]["input_size"]
    fold = config["train"]["fold"]
    shot = config["train"]["shot"]
    train_snapshot = config["train"]["train_snapshot"]
    test_snapshot = config["train"]["test_snapshot"]
    num_class = config["train"]["num_class"]
    bs = config["train"]["batch_size"]
    nw = config["train"]["num_workers"]
    snapshot = {"train": train_snapshot, "test": test_snapshot}

    trainset = FewShot(img_path, mask_path, crop_size, fold, shot, snapshot["train"], num_class, mode="train")
    trainloader = DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True, drop_last=True)

    testset = FewShot(img_path, mask_path, crop_size, fold, shot, snapshot["test"], num_class, mode="val")
    testloader = DataLoader(testset, batch_size=1, shuffle=False, pin_memory=True, num_workers=nw, drop_last=False)

    # device
    device = config["train"]["device"]
    device = select_device(device, batch_size=bs)

    # model
    model_name = config["train"]["model_name"]
    refine = config["train"]["refine"]
    model = SSPNet(model_name, refine=refine)
    logging.info(model)
    logging.info('\nModel Params: %.1fM' % count_params(model))

    # load weights
    weights = config["train"]["weights"]
    if weights is not None and os.path.exists(weights):
        # 加载模型权重
        logging.info(f"Load pretrain weights from {weights}")
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model.load_state_dict(ckpt, strict=False)  # load

    for param in model.layer0.parameters():
        param.requires_grad = False
    for param in model.layer1.parameters():
        param.requires_grad = False
    # for param in model.layer2.parameters():
    #    param.requires_grad = False
    # for name, param in model.named_parameters():
    #    print(name, param.requires_grad)

    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            for param in module.parameters():
                param.requires_grad = False

    model = torch.nn.DataParallel(model).to(device)
    # model = model.to(device)
    best_model = None

    # optimizer
    lr = config["train"]["lr"]
    params = [p for p in model.parameters() if p.requires_grad]
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)

    iters = 0
    total_iters = config["train"]["episode"] // bs
    lr_decay_iters = [total_iters // 3, total_iters * 2 // 3]
    previous_best = 0

    # each snapshot is considered as an epoch
    for epoch in range(config["train"]["episode"] // train_snapshot):
        logging.info("\n==> Epoch %i, learning rate = %.5f\t\t\t\t Previous best = %.2f"
                     % (epoch, optimizer.param_groups[0]["lr"], previous_best))
        model.train()

        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()

        total_loss = 0.0

        tbar = tqdm(trainloader)
        set_seed(int(time.time()))

        for i, (img_s_list, mask_s_list, img_q, mask_q, cls, id_s_list, id_q) in enumerate(tbar):
            img_q, mask_q = img_q.to(device), mask_q.to(device)
            for k in range(len(img_s_list)):
                img_s_list[k], mask_s_list[k] = img_s_list[k].to(device), mask_s_list[k].to(device)

            out_list = model(img_s_list, mask_s_list, img_q, mask_q)
            mask_s = torch.cat(mask_s_list, dim=0)

            if refine:
                loss = criterion(out_list[0], mask_q) + \
                       criterion(out_list[1], mask_q) + \
                       criterion(out_list[2], mask_q) + \
                       criterion(out_list[3], mask_s) * 0.2
            else:
                loss = criterion(out_list[0], mask_q) + \
                       criterion(out_list[1], mask_q) + \
                       criterion(out_list[2], mask_s) * 0.2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            iters += 1
            if iters in lr_decay_iters:
                optimizer.param_groups[0]['lr'] /= 10.0

            tbar.set_description(
                'Iter: %.3d,Loss: %.3f Lr: %.5f' % (iters, total_loss / (i + 1), optimizer.param_groups[0]['lr']))
            tb_writer.add_scalar("Loss", total_loss / (i + 1), iters)
            tb_writer.add_scalar("Lr", optimizer.param_groups[0]["lr"], iters)

        model.eval()
        set_seed(0)
        logging.info(
            'Epoch: %.d Loss: %.3f lr: %.5f' % (epoch, total_loss / train_snapshot, optimizer.param_groups[0]["lr"]))

        miou = evaluate(model, testloader, device, config)

        if miou >= previous_best:
            best_model = deepcopy(model)
            previous_best = miou
            torch.save(best_model.module.state_dict(),
                       os.path.join(str(weights_dir), '%s_%ishot_%iepoch_%.2f.pth' % (model_name, shot, epoch, miou)))

    logging.info('\nEvaluating on 5 seeds.....')
    total_miou = 0.0
    for seed in range(5):
        logging.info('\nRun %i:' % (seed + 1))
        set_seed(seed)

        miou = evaluate(best_model, testloader, device, config)
        total_miou += miou

    logging.info('\n' + '*' * 32)
    logging.info('Averaged mIOU on 5 seeds: %.2f' % (total_miou / 5))
    logging.info('*' * 32 + '\n')

    torch.save(best_model.module.state_dict(),
               os.path.join(str(weights_dir), '%s_%ishot_%.2f.pth' % (model_name, shot, total_miou / 5)))


if __name__ == '__main__':
    config = parser_args()
    main(config)
