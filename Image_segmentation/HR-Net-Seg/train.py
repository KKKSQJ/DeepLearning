import logging
import time
import os
import yaml
import shutil
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_ema import ExponentialMovingAverage

from dataLoader.dataloader import build_loaders
from models.network import build_model
from utils.utils import build_optim, seed_everything, add_to_logs, AverageMeter, copy_parameters_to_model, \
    copy_parameters_from_model, create_lr_scheduler
from utils.evaluate import Evaluator
from loss.dice_score import dice_loss


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


def run(hyperparams):
    print(hyperparams)

    amp = hyperparams['train']['amp']
    scaler = torch.cuda.amp.GradScaler() if amp else None

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create model, loaders, optimizer, etc
    in_channel = hyperparams["model"]["extra"]["in_channel"]
    num_classes = hyperparams["dataset"]["num_classes"] + 1
    model = build_model(in_channel=in_channel, num_classes=num_classes, config=hyperparams).to(device)
    loader = build_loaders(hyperparams, mode='train')

    ema = hyperparams['train']['ema']
    ema_decay_per_epoch = hyperparams['train']['ema_decay_per_epoch']
    if ema:
        iters = len(loader["train_loader"])
        ema_decay = ema_decay_per_epoch ** (1 / iters)
        ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)

    ignore_label = hyperparams["criterion"]["params"]["ignore_label"]
    optimizer_params = hyperparams["optimizer"]
    scheduler_params = hyperparams["scheduler"]
    criterion_params = hyperparams["criterion"]

    criterion_params["params"]["ignore_label"] = ignore_label
    criterion_params["params"]["weight"] = loader["train_dataset"].dataset.class_weights if \
        hyperparams["dataset"]["val"][
            "image_path"] is None else loader["train_dataset"].class_weights

    optim = build_optim(model, optimizer_params, scheduler_params, criterion_params)
    criterion, optimizer, scheduler = (
        optim["criterion"],
        optim["optimizer"],
        optim["scheduler"],
    )
    if scheduler is None:
        scheduler = create_lr_scheduler(optimizer, loader["train_loader"].__len__(), hyperparams["train"]["n_epochs"],
                                        warmup=True)

    # logging
    logging_name = hyperparams['train']['logging_name']
    if logging_name is None:
        logging_name = "model_{}_dataset_{}_loss_{}".format(hyperparams["model"]["name"],
                                                            hyperparams["dataset"]["name"],
                                                            hyperparams["criterion"]["name"])
    if not hyperparams["train"]["resume"]:
        shutil.rmtree("weights/{}".format(logging_name), ignore_errors=True)

    shutil.rmtree(
        "runs/{}".format(logging_name),
        ignore_errors=True,
    )
    shutil.rmtree(
        "logs/{}".format(logging_name),
        ignore_errors=True,
    )
    os.makedirs(
        "logs/{}".format(logging_name),
        exist_ok=True,
    )

    writer = SummaryWriter("runs/{}".format(logging_name))
    logging_dir = "logs/{}".format(logging_name)
    logging_path = os.path.join(logging_dir, "train.log")
    logging.basicConfig(filename=logging_path, level=logging.INFO, filemode="w+")

    best_mIoU = 0
    last_epoch = 0
    best_epoch = 0

    if hyperparams["train"]["resume"]:
        model_state_file = "weights/{}/best.pth".format(logging_name)
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location=device)
            best_mIoU = checkpoint["best_mIoU"]
            last_epoch = checkpoint["epoch"]
            dct = checkpoint["state_dict"]
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint["lr_scheduler"])
            add_to_logs(logging, "=> loaded checkpoint (epoch {})".format(checkpoint["epoch"]))

    n_epochs = hyperparams["train"]["n_epochs"]
    end_epoch = last_epoch + n_epochs
    epoch_iters = loader["train_loader"].__len__()  # / hyperparams["train"]["batch_size"]
    global_step = last_epoch * epoch_iters

    for epoch in range(last_epoch, end_epoch):
        # train
        add_to_logs(logging, "{}, epoch {}".format(time.ctime(), epoch))

        model.train()

        batch_time = AverageMeter()
        ave_loss = AverageMeter()
        tic = time.time()

        with tqdm(total=loader["train_dataset"].__len__(), desc=f'Epoch {epoch}/{end_epoch}',
                  unit='img') as pbar:
            for i_iter, batch in enumerate(loader["train_loader"]):
                images, labels, _, _ = batch
                images = images.to(device)
                labels = labels.long().to(device)

                if scaler:
                    with torch.cuda.amp.autocast(enabled=amp):
                        masks_pred = model(images)
                        if masks_pred.size() != labels.size():
                            masks_pred = F.interpolate(masks_pred, size=labels.size()[-2:], mode='bilinear',
                                                       align_corners=False)
                        loss = criterion(masks_pred, labels) \
                               + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                           F.one_hot(labels, num_classes).permute(0, 3, 1, 2).float(),
                                           multiclass=True)
                        # loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_label,
                        #                                  weight=loader["train_dataset"].class_weights)(masks_pred,
                        #                                                                                labels) \
                        #        + dice_loss(F.softmax(masks_pred, dim=1).float(),
                        #                    F.one_hot(labels, num_classes).permute(0, 3, 1, 2).float(),
                        #                    multiclass=True)
                else:
                    masks_pred = model(images)
                    if masks_pred.size() != labels.size():
                        masks_pred = F.interpolate(masks_pred, size=labels.size()[-2:], mode='bilinear',
                                                   align_corners=False)

                    loss = criterion(masks_pred, labels) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(labels, num_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)
                    # loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_label,
                    #                                  weight=loader["train_dataset"].class_weights)(masks_pred,
                    #                                                                                labels) \
                    #        + dice_loss(F.softmax(masks_pred, dim=1).float(),
                    #                    F.one_hot(labels, num_classes).permute(0, 3, 1, 2).float(),
                    #                    multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                if ema:
                    ema.update(model.parameters())

                # scheduler.step()
                pbar.update(images.shape[0])
                global_step += 1

                batch_time.update(time.time() - tic)
                tic = time.time()

                ave_loss.update(loss.item())

                writer.add_scalar("train loss", loss.item(), global_step)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                if i_iter % hyperparams["train"]["print_freq"] == 0:
                    msg = 'train Epoch: [{}/{}] Iter:[{}/{}], Time: {:.2f}, ' \
                          'lr: {}, Loss: {:.6f}'.format(
                        epoch, end_epoch, i_iter, epoch_iters,
                        batch_time.average(), [x['lr'] for x in optimizer.param_groups],
                        ave_loss.average())
                    add_to_logs(logging, msg)

        if ema:
            copy_of_model_parameters = copy_parameters_from_model(model)
            ema.copy_to(model.parameters())

        # eval
        if epoch % hyperparams["train"]["eval_step"] == 0 or epoch == end_epoch - 1:
            valid_time = AverageMeter()
            t = time.time()
            model.eval()
            metric = Evaluator(num_class=num_classes)
            valid_loss = AverageMeter()
            with torch.no_grad():
                for idx, batch in enumerate(loader["val_loader"]):
                    image, label, _, _ = batch
                    size = label.size()
                    image = image.to(device)
                    label = label.long().to(device)

                    pred = model(image)

                    if pred.size() != label.size():
                        pred = F.interpolate(input=pred, size=size[-2:], mode='bilinear', align_corners=False)

                    loss = criterion(pred, label) \
                           + dice_loss(F.softmax(pred, dim=1).float(),
                                       F.one_hot(label, num_classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)
                    # loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_label,
                    #                                  weight=loader["val_dataset"].class_weights)(pred,
                    #                                                                              label) \
                    #        + dice_loss(F.softmax(pred, dim=1).float(),
                    #                    F.one_hot(label, num_classes).permute(0, 3, 1, 2).float(),
                    #                    multiclass=True)

                    metric.add_batch(label.flatten(), torch.softmax(pred, dim=1).argmax(1).flatten())
                    acc = metric.Pixel_Accuracy()
                    mIoU = metric.Mean_Intersection_over_Union()
                    add_to_logs(logging, "epoch: {} idx: {} Pixel Acc:{} mIOU:{}".format(epoch, idx, acc, mIoU))
                    metric.reset()

                    valid_time.update(time.time() - t)
                    t = time.time()
                    valid_loss.update(loss.item())
                    writer.add_scalar("valid loss", valid_loss.average(), epoch)
                    writer.add_scalar("valid pixel ac", acc, epoch)
                    writer.add_scalar("valid miou", mIoU, epoch)
                    writer.add_image("images", image[0].cpu(), epoch)
                    writer.add_image('mask_true', label[0][None, :].float().cpu(), epoch)
                    writer.add_image('mask_pred',
                                     torch.softmax(pred, dim=1).argmax(dim=1)[0][None, :].float().cpu(), epoch)

            msg = 'valid Epoch: [{}/{}] Time: {:.2f}, ' \
                  'lr: {}, Loss: {:.6f}'.format(
                epoch, end_epoch, valid_time.average(), [x['lr'] for x in optimizer.param_groups],
                valid_loss.average())
            add_to_logs(logging, msg)

        if mIoU > best_mIoU:
            add_to_logs(logging,
                        "Loss: {:.3f}\n"
                        "Pixel: {:.3f}\n"
                        "MIoU increased ({:.3f} --> {:.3f})\n. Saving model ...".format(
                            valid_loss.average(),
                            acc,
                            best_mIoU, mIoU))
            best_mIoU = mIoU
            best_epoch = epoch

            os.makedirs(
                "weights/{}".format(logging_name),
                exist_ok=True,
            )

            save_file = {"state_dict": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "lr_scheduler": scheduler.state_dict(),
                         "epoch": epoch,
                         "best_mIoU": best_mIoU
                         }
            if amp:
                save_file["scaler"] = scaler.state_dict()

            torch.save(save_file, "weights/{}/epoch{}.pth".format(logging_name, epoch))
            shutil.copy("weights/{}/epoch{}.pth".format(logging_name, epoch),
                        "weights/{}/best.pth".format(logging_name))

        add_to_logs(logging, "best epoch: {}".format(best_epoch))

        if ema:
            copy_parameters_to_model(copy_of_model_parameters, model)
        scheduler.step()

    writer.close()


if __name__ == '__main__':
    # Usage
    """
    python train.py --config_name config/example.yaml
    """
    hyperparams = parse_config()
    run(hyperparams)
