import os
import math
import argparse
import numpy as np
import shutil

import torch
import torchvision
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler

from data_utils import read_split_data, matplotlib_imshow, plot_data_loader_image, plot_class_preds
from dataset import DataSet
from model import resnet34
from train_eval_utils import train_one_epoch, evaluate

def main(args):
    print(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    # 实例化SummaryWriter对象
    tb_writer = SummaryWriter(log_dir="runs")
    assert len(args.out_path) != 0, "请设置模型权重的输出路径"
    if os.path.exists(args.out_path) is False:
        os.makedirs(args.out_path)

    # 准备训练数据，并且划分为训练集和验证集
    train_data, val_data, every_class_num = read_split_data(args.data_path)

    # 定义训练以及预测时的预处理方法
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_data_set = DataSet(data=train_data, transform=data_transform["train"])
    # 实例化验证数据集
    val_data_set = DataSet(data=val_data, transform=data_transform["val"])

    # 计算使用num_workers的数量
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    # 实例化训练集data loader
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_data_set.collate_fn)
    # 实例化验证集data loader
    val_loader = torch.utils.data.DataLoader(val_data_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_data_set.collate_fn)

    # 将图片写入tensorboard
    # 随机获取一些训练图片
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    # 如果需要将多张图片合并为一张图片，使用torchvision.utils.make_grid(images)
    img_grid = torchvision.utils.make_grid(images,padding=0)
    # matplotlib_imshow()对img_grid进行处理，输出归一化的图片以及原始图片
    norm_image, unnorm_image = matplotlib_imshow(img_grid,one_channel=False)
    tb_writer.add_image("归一化后的图片",norm_image)
    tb_writer.add_figure("原始图片", unnorm_image,global_step=0)


    # 实例化模型
    model = resnet34(num_classes=args.num_classes).to(device)

    # images = torch.zeros((1,3,224,224),device=device)
    # 将模型写入tensorboard
    # 注意：如果模型加载到了cuda,那么数据也要加载到cuda，否则会报错
    tb_writer.add_graph(model, images.to(device))

    # 如果存在预训练权重则载入
    if os.path.exists(args.weights):
        weight_dict = torch.load(args.weights, map_location=device)
        load_weights_dict = {k: v for k, v in weight_dict.items()
                             if model.state_dict()(k).numel() == v.numel()}
        model.load_state_dict(load_weights_dict, strict=False)   # strict=False,不完全匹配
    else:
        print("not using pretrain-weights.")

    # 是否冻结权重
    if args.freeze_layers:
        print("freeze layers except fc layer.")
        for name, para in model.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg,lr=args.lr, momentum=0.9, weight_decay=0.005)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


    best_epoch = 0
    best_acc = -np.inf
    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        # 更新学习率
        scheduler.step()

        # val
        val_loss, val_acc = evaluate(model=model,
                                    data_loader=val_loader,
                                    device=device,
                                    epoch=epoch)

        # 将loss,acc,lr添加到tensorboard
        print("[epoch {}] train loss: {}".format(epoch, round(train_loss, 3)))
        print("[epoch {}] train accuracy: {}".format(epoch, round(train_acc, 3)))
        print("[epoch {}] val loss: {}".format(epoch, round(val_loss, 3)))
        print("[epoch {}] val accuracy: {}".format(epoch, round(val_acc, 3)))

        tags = ["train_loss", "train_accuracy", "val_loss", "val_accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4],optimizer.param_groups[0]["lr"], epoch)

        # 将模型在验证集上的验证结果(图片加预测的类别)添加到tensorboard
        fig = plot_class_preds(net=model,
                               images_dir="./plot_img",
                               transform=data_transform["val"],
                               num_plot=5,
                               device=device)

        if fig is not None:
            tb_writer.add_figure("predictions vs. actuals",
                                 figure=fig,
                                 global_step=epoch)

        # add conv1 weights into tensorboard
        tb_writer.add_histogram(tag="conv1",
                                values=model.conv1.weight,
                                global_step=epoch)
        tb_writer.add_histogram(tag="layer1/block0/conv1",
                                values=model.layer1[0].conv1.weight,
                                global_step=epoch)

        # save weights
        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
        if val_loss > best_acc:
            best_epoch = epoch
            best_acc = val_acc
            shutil.copy("./weights/model-{}.pth".format(best_epoch), "./weights/best_mdoel.pth")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1)

    # 数据集
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str, default='./data')

    # resnet34预训练权重
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    parser.add_argument('--weights', type=str, default='resnet34.pth', help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out-path', type=str, default='./weights')

    opt = parser.parse_args()
    main(opt)
