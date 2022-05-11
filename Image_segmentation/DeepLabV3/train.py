import os
import time
import datetime
import logging
import argparse
import yaml
import shutil
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from dataLoader.transforms import build_transform
from dataLoader.dataloader import build_dataloader
from models.network import build_model
from utils.utils import build_optim, create_lr_scheduler, lr_scheduler, seed_everything, AverageMeter
from utils.confusion_matrix import ConfusionMatrix


def run(hyperparams):
    print(hyperparams)
    seed_everything()

    # Create logging
    logging_name = hyperparams['train']['logging_name']
    if logging_name is None:
        logging_name = "model_{}_dataset_{}_loss_{}".format(hyperparams["model"]["name"],
                                                            hyperparams["dataset"]["type"],
                                                            hyperparams["criterion"]["name"])
    shutil.rmtree("runs/{}".format(logging_name), ignore_errors=True, )
    shutil.rmtree("logs/{}".format(logging_name), ignore_errors=True, )
    os.makedirs("logs/{}".format(logging_name), exist_ok=True, )

    writer = SummaryWriter("runs/{}".format(logging_name))
    logging_dir = "logs/{}".format(logging_name)
    logging_path = os.path.join(logging_dir, "train.log")
    logging.basicConfig(filename=logging_path, level=logging.INFO, filemode="w+")
    for k, v in hyperparams.items():
        logging.info(f"====>  {k}: {v}   <=====")

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() and hyperparams["train"]["use_gpu"] else "cpu")
    logging.info(f"====>  Train device: {device}   <=====")
    num_classes = hyperparams["train"]["num_classes"] + 1
    logging.info(f"====>  Num classes: {num_classes}   <=====")

    model = build_model(hyperparams, num_classes=num_classes, pretrain=True)
    model = model.to(device)
    logging.info(f"====>  Init model successful: {model}   <=====")

    # Create transforms and dataloader
    trans_train = build_transform(hyperparams, train=True)
    trans_val = build_transform(hyperparams, train=False)
    loader = build_dataloader(hyperparams, transform={"train": trans_train, "val": trans_val})
    train_loader, val_loader = loader["train"], loader["val"]
    logging.info(
        f"====>  Init train loader successful, size: {len(train_loader)} x batch size: {hyperparams['train']['batch_size']} <=====")
    logging.info(
        f"====>  Init val loader successful, size: {len(val_loader)} x batch size: {hyperparams['train']['batch_size']}  <=====")

    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    ]

    if hyperparams["model"]["use_aux"]:
        params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": hyperparams["optimizer"]["params"]["lr"] * 10})

    # Create optimizer, critter,
    optimizer_params = hyperparams["optimizer"]
    criterion_params = hyperparams["criterion"]
    scheduler_params = hyperparams["scheduler"]
    scheduler_params["params"]["lr_lambda"] = lr_scheduler(len(train_loader), hyperparams["train"]["epochs"])
    optim = build_optim(params_to_optimize, optimizer_params, criterion_params, scheduler_params)
    criterion, optimizer, scheduler = (
        optim["criterion"],
        optim["optimizer"],
        optim["scheduler"],
    )

    # if scheduler is None:
    # scheduler = create_lr_scheduler(optimizer, len(train_loader), hyperparams["train"]["epochs"], warmup=True)

    # import matplotlib.pyplot as plt
    # lr_list = []
    # for _ in range(hyperparams["train"]["epochs"]):
    #     for _ in range(len(train_loader)):
    #         scheduler.step()
    #         lr = optimizer.param_groups[0]["lr"]
    #         lr_list.append(lr)
    # plt.plot(range(len(lr_list)), lr_list)
    # plt.show()

    # Use torch.cuda.amp for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if hyperparams["train"]["amp"] else None

    start_epoch = hyperparams["train"]["start_epoch"]
    max_epochs = hyperparams["train"]["epochs"]

    if hyperparams["train"]["resume"]:
        checkpoint = torch.load(hyperparams["train"]["resume"], map_location='cpu')
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        if hyperparams["train"]["amp"]:
            scaler.load_state_dict(checkpoint["scaler"])
        logging.info(f"====>  Resume from checkpoint: {hyperparams['train']['resume']}   <=====")

    start_time = time.time()
    end_epoch = start_epoch + max_epochs
    global_step = len(train_loader) * start_epoch
    eval_global_step = len(val_loader) * start_epoch
    best_mIoU = 0.0
    best_epoch = -1

    for epoch in range(start_epoch, end_epoch):
        logging.info(f"====>  {time.ctime()},epoch {epoch}  <=====")

        # train
        model.train()

        batch_time = AverageMeter()
        ave_loss = AverageMeter()
        tic = time.time()

        with tqdm(total=len(train_loader) * hyperparams["train"]["batch_size"], desc=f'Epoch {epoch}/{end_epoch}',
                  unit='img') as pbar:
            for i_iter, batch in enumerate(train_loader):
                images, targets = batch
                images = images.to(device)
                targets = targets.to(device)

                losses = {}
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    output = model(images)
                    for name, x in output.items():
                        losses[name] = criterion(x, targets)
                    if len(losses) == 1:
                        loss = losses["out"]
                    else:
                        loss = losses["out"] + 0.5 * losses["aux"]

                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                scheduler.step()
                pbar.update(images.shape[0])
                global_step += 1

                batch_time.update(time.time() - tic)
                tic = time.time()

                ave_loss.update(loss.item())

                lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("train loss", loss.item(), global_step)
                writer.add_scalar("lr", lr, global_step)
                pbar.set_postfix(**{"(batch) loss": loss.item(), "lr": lr})

                if i_iter % hyperparams["train"]["print_freq"] == 0:
                    msg = 'train Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                          'lr: {}, Loss: {:.6f}'.format(
                        epoch, end_epoch, i_iter, len(train_loader),
                        batch_time.average(), [x['lr'] for x in optimizer.param_groups],
                        ave_loss.average())

                    logging.info(msg)

        # eval
        if epoch % hyperparams["train"]["eval_step"] == 0 or epoch == end_epoch - 1:
            valid_time = AverageMeter()
            valid_loss = AverageMeter()
            t = time.time()
            model.eval()
            confmat = ConfusionMatrix(num_classes)

            with torch.no_grad():
                with tqdm(total=len(val_loader) * hyperparams["train"]["batch_size"],
                          desc=f'Epoch {epoch}/{end_epoch}',
                          unit='img') as pbar:
                    for i_iter, batch in enumerate(val_loader):
                        images, targets = batch
                        images = images.to(device)
                        targets = targets.to(device)

                        output = model(images)
                        output = output["out"]

                        confmat.update(targets.flatten(), output.argmax(1).flatten())

                        pbar.update(images.shape[0])

                        valid_loss.update(loss.item())
                        if i_iter % hyperparams["train"]["print_freq"] == 0:
                            writer.add_scalar("valid loss", valid_loss.average(), eval_global_step)
                            writer.add_image("images", images[0].cpu(), eval_global_step)
                            writer.add_image('mask_true', targets[0][None, :].float().cpu(), eval_global_step)
                            writer.add_image('mask_pred',
                                             torch.softmax(output, dim=1).argmax(dim=1)[0][None, :].float().cpu(),
                                             eval_global_step)
                        eval_global_step += 1

                    confmat.reduce_from_all_processes()
            val_info = str(confmat)
            print(val_info)
            acc_global, acc, iu = confmat.get_result()
            mIoU = iu.mean().item()

        # save mode
        if mIoU > best_mIoU:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {ave_loss.average():.4f}\n" \
                         f"lr: {lr:.6f}\n"
            logging.info(train_info)
            val_info = str(confmat)
            logging.info(val_info)

            best_mIoU = mIoU
            best_epoch = epoch

            os.makedirs(
                "weights/{}".format(logging_name),
                exist_ok=True,
            )

            save_file = {"model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(),
                         "epoch": epoch,
                         "best_mIoU": best_mIoU}

            if hyperparams["train"]["amp"]:
                save_file["scaler"] = scaler.state_dict()

            torch.save(save_file, "weights/{}/epoch{}.pth".format(logging_name, epoch))
            shutil.copy("weights/{}/epoch{}.pth".format(logging_name, epoch),
                        "weights/{}/best.pth".format(logging_name))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))
    logging.info("training time {}".format(total_time_str))
    writer.close()


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_name",
        type=str,
        default="config/example.yaml",
    )

    parser_args = parser.parse_args()

    with open(vars(parser_args)["config_name"], "r", encoding='utf-8') as config_file:
        hyperparams = yaml.full_load(config_file)

    return hyperparams


if __name__ == '__main__':
    # Usage
    """
    python train.py --config_name config/example.yaml
    """
    hyperparams = parse_config()
    run(hyperparams)
