import math
import shutil

import torch
import torch.nn as nn
import torchvision
from torchvision.models import AlexNet, resnet50, resnet18
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import os
import sys
import argparse
from tqdm import tqdm
import numpy as np

from models.network import mnist_cnn, mnist_fcn
from dataLoader.dataSet import read_split_data
from dataLoader.dataLoader import mnist_Dataset
from utils import matplotlib_imshow, train_one_epoch, evaluate


def main(opt):
    print(opt)

    assert os.path.exists(opt.data_path), "The Mnist data path:{} does not exists".format(opt.data_path)

    tb_writer = SummaryWriter()
    save_dir = tb_writer.log_dir
    weights_dir = save_dir + "/weights"
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    device = torch.device("cuda" if torch.cuda.is_available() and opt.use_cuda else "cpu")

    # 读取数据
    train_images_path, val_images_path, train_images_label, val_images_label, every_class_num = read_split_data(
        opt.data_path, save_dir, val_rate=0.2, plot_image=True)

    data_transform = {
        "train": transforms.Compose([  # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        "val": transforms.Compose([  # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])}

    # 实例化训练数据集
    train_dataset = mnist_Dataset(images_path=train_images_path,
                                  images_class=train_images_label,
                                  transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = mnist_Dataset(images_path=val_images_path,
                                images_class=val_images_label,
                                transform=data_transform["val"])

    batch_size = opt.batch_size
    nw = min([os.cpu_count(), batch_size, opt.num_worker if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                               pin_memory=True, num_workers=nw, collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                             num_workers=nw, collate_fn=val_dataset.collate_fn)
    num_classes = len(every_class_num)

    # 将图片写入tensorboard
    # 随机获取一些训练图片
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    # 如果需要将多张图片合并为一张图片，使用torchvision.utils.make_grid(images)
    img_grid = torchvision.utils.make_grid(images, padding=0)
    # matplotlib_imshow()对img_grid进行处理，输出归一化的图片以及原始图片
    norm_image, unnorm_image = matplotlib_imshow(img_grid, one_channel=False)
    tb_writer.add_image("归一化后的图片", norm_image)
    tb_writer.add_figure("原始图片", unnorm_image, global_step=0)

    # 初始化网络
    # 使用自定义网络
    # model = mnist_cnn(num_classes=num_classes).to(device)

    model = mnist_fcn(num_classes=num_classes).to(device)
    #
    # # 使用内置网络
    # # 改变全连接层的方法，法1：
    # model = resnet18(num_classes=num_classes).to(device)
    # if os.path.exists(opt.weight):
    #     model.load_state_dict(torch.load(opt.weight, map_location=device), strict=False)
    #
    # # 法2：
    # model = resnet18().to(device)
    # in_channel = model.fc.in_features
    # model.fc = nn.Linear(in_channel, num_classes)
    # if os.path.exists(opt.weight):
    #     model.load_state_dict(torch.load(opt.weight, map_location=device), strict=False)

    if device.type == 'cuda':
        graph_inputs = torch.from_numpy(np.random.rand(1, 3, 28, 28)).type(
            torch.FloatTensor).cuda()
    else:
        graph_inputs = torch.from_numpy(np.random.rand(1, 3, 28, 28)).type(torch.FloatTensor)
    tb_writer.add_graph(model, (graph_inputs,))

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        del_keys = ['fc.weight', 'fc.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除fc外，其他权重全部冻结
            if "fc" in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    if opt.optimizer == 'SGD':
        optimizer = torch.optim.SGD(pg, lr=opt.lr, momentum=0.9, weight_decay=5E-5)
    else:
        optimizer = torch.optim.Adam(pg, lr=opt.lr, weight_decay=1e-3)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / opt.epochs)) / 2) * (1 - opt.lrf) + opt.lrf  # cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 20], 0.1)
    loss_function = torch.nn.CrossEntropyLoss()

    best_acc = -np.inf
    best_epoch = 0
    for epoch in range(opt.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model, train_loader, device, optimizer, loss_function, epoch)
        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model, val_loader, device, loss_function, epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate", "images"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        batch_images = next(iter(train_loader))[0]
        tb_writer.add_images(tags[5], batch_images, epoch)

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            best_epoch = epoch + 1
        model_path = weights_dir + "/model_{}.pth".format(epoch)
        torch.save(model.state_dict(), model_path)
        if is_best:
            shutil.copy(model_path, weights_dir + "/best_model.pth")
    tb_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 数据集
    parser.add_argument('--data-path', type=str, default='./data', help='The Mnist data path')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-worker', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.01)

    # resnet34预训练权重
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    parser.add_argument('--weights', type=str, default='', help='initial weights path')  # resnet18.pth
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--use_cuda', default=True)
    parser.add_argument('--optimizer', type=str, default='SGD')  # Adam

    args = parser.parse_args()

    main(args)
