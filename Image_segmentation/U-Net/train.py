import argparse
import os.path
import numpy as np
from loguru import logger
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models import UNet
from dataLoader import CarvanaDataset, BasicDataset
from utils import setup_logger
from loss import dice_loss
from evaluate import evaluate


@logger.catch
def run(args):
    tensorboard = SummaryWriter()
    setup_logger(
        save_dir=tensorboard.log_dir,
        distributed_rank=0,
        filename='train_log.txt',  # 'train_log_{time}.txt',
        mode='a'
    )
    logger.info("===========user config=============")
    # logger.info("Args: {}".format(self.args))
    for k, v in args.__dict__.items():
        logger.info("{:}:{:}".format(k, v))
    logger.info("=============end===================")
    dir_checkpoint = tensorboard.log_dir + '/checkpoint'

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    logger.info(f"Using device {device}")

    # 0. Create model
    """
    Change here to adapt to your data
    in_channels=3 for RGB images
    classes is the number of probabilities you want to get per pixel
    billinear:different Upsample op
    """
    model = UNet(in_channel=3, classes=2, bilinear=args.bilinear)

    logger.info(f'Network:\n'
                f'\t{model.in_channel} input channels\n'
                f'\t{model.classes} output channels (classes)\n'
                f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device))
        logger.info(f'Model loaded from {args.load}')

    model.to(device)

    if device.type == 'cuda':
        graph_inputs = torch.from_numpy(np.random.rand(1, 3, 224, 224)).type(
            torch.FloatTensor).cuda()
    else:
        graph_inputs = torch.from_numpy(np.random.rand(1, 3, 224, 224)).type(torch.FloatTensor)
    tensorboard.add_graph(model, (graph_inputs,))

    # 1. Create data
    """
    imgs_dir: path of images dir.
    mask_dir: path of masks dir.
    scale: Downscaling factor of the images.
    """
    dataset = BasicDataset(args.imgs_dir, args.masks_dir, args.scale)
    # try:
    #     dataset = CarvanaDataset(args.imgs_dir, args.masks_dir, args.scale)
    # except(AssertionError, RuntimeError):
    #     dataset = BasicDataset(args.imgs_dir, args.masks_dir, args.scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * args.validation)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=args.batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    logger.info(f'''Starting training:
        Epochs:          {args.epochs}
        Batch size:      {args.batch_size}
        Learning rate:   {args.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {args.save_checkpoint}
        Device:          {device.type}
        Images scaling:  {args.scale}
        Mixed Precision: {args.amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == model.in_channel, \
                    f'Network has been defined with {model.in_channel} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=args.amp):
                    masks_pred = model(images)
                    loss = criterion(masks_pred, true_masks) \
                           + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                       F.one_hot(true_masks, model.classes).permute(0, 3, 1, 2).float(),
                                       multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                tensorboard.add_scalar("train loss", loss.item(), global_step)
                tensorboard.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (10 * args.batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            tensorboard.add_histogram('Weights/' + tag, value.data.cpu(), division_step)
                            tensorboard.add_histogram('Gradients/' + tag, value.grad.data.cpu(), division_step)

                        val_score = evaluate(model, val_loader, device)
                        scheduler.step(val_score)

                        logger.info('Step:{}, Validation Dice score: {}'.format(division_step,val_score))

                        tensorboard.add_scalar('validation Dice', val_score, division_step)
                        tensorboard.add_image("images", images[0].cpu(), division_step)
                        tensorboard.add_figure('mask_true', true_masks[0][None,:].float().cpu())
                        tensorboard.add_figure('mask_pred',
                                               torch.softmax(masks_pred, dim=1).argmax(dim=1)[0][None,:].float().cpu())
        logger.info("epoch: {} epoch loss: {}".format(epoch, epoch_loss))
        if args.save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(dir_checkpoint,'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logger.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--imgs_dir', type=str, default='./data/imgs', help='Path of img')
    parser.add_argument('--masks_dir', type=str, default='./data/mask', help='Path of mask')
    parser.add_argument('--save_checkpoint', type=bool, default=True, help='Store the model weights')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', type=float, default=0.1,
                        help='Percent of the data that is used as validation (0-1)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--use_cuda', action='store_true', default=True, help='Use cuda to train')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    run(args)
