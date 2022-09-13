import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utils import accuracy, ProgressMeter, AverageMeter
from models import model_dict,get_model
from torch.utils.data import DataLoader
from torchvision import transforms
from dataLoader import My_Dataset_with_txt


def parser_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Test')
    parser.add_argument('--data', metavar='DIR', help='path to dataset',required=True)
    parser.add_argument('--classes', metavar='CLASSES', default=5)
    parser.add_argument('--weights', metavar='WEIGHTS', help='path to the weights file', required=True)
    parser.add_argument('-a', '--arch', metavar='ARCH', default='shufflenet_v1_g3')
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=100, type=int,
                        metavar='N',
                        help='mini-batch size (default: 100) for test')
    parser.add_argument('-r', '--resolution', default=224, type=int,
                        metavar='R',
                        help='resolution (default: 224) for test')

    args = parser.parse_args()
    return args


def test(args):
    assert args.arch in model_dict
    model_build_func = get_model(args.arch)
    model = model_build_func(num_classes=args.classes)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
        use_gpu = False
    else:
        model = model.cuda()
        use_gpu = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if os.path.isfile(args.weights):
        print("=> loading checkpoint '{}'".format(args.weights))
        model.load_state_dict(torch.load(args.weights, map_location="cuda:0")["state_dict"], strict=True)
    else:
        print("=> no checkpoint found at '{}'".format(args.weights))

    cudnn.benchmark = True

    transform = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    val_set = My_Dataset_with_txt(args.data, "val.txt", transform=transform)
    val_dataloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=True, drop_last=False,
                                collate_fn=val_set.collate_fn)

    validate(val_dataloader, model, criterion, use_gpu)


def validate(val_loader, model, criterion, use_gpu):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if use_gpu:
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            if isinstance(output,
                          dict):  # If the model being tested is a training-time RepVGGplus, which has auxiliary classifiers
                output = output['main']
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


if __name__ == '__main__':
    args = parser_args()
    test(args)
