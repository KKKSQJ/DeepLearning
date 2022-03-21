import argparse
import json
import math
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import posixpath
import os
from tqdm import tqdm

from data_utils import data_reader, preprocessing
from models.MadNet import madnet
from losses import loss_factory
from Sampler import sampler_factory

import torch
import torch.nn as nn
from torchvision import transforms

MAX_DISP = 255
PIXEL_TH = 3


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_file', default='./data_utils/file_path.txt', type=str,
                        help='path to the list file with frames to be processed')

    parser.add_argument('--blockConfig', default='block_config/MadNet_piramid.json', type=str,
                        help='')

    parser.add_argument('--output', default='result', type=str,
                        help='disparity output path')

    parser.add_argument('--epochs', default=1, type=int,
                        help='max epochs')

    parser.add_argument('--crop_shape', default=(320, 1216),
                        help='two int for the size of the crop extracted from each image [height,width]')

    parser.add_argument('--mode', default='FULL',
                        help='online adaptation mode: NONE - perform only inference, FULL - full online backprop, MAD - backprop only on portions of the network')

    parser.add_argument('--lr', default=0.0001, type=float,
                        help='value for learning rate')

    parser.add_argument('--device', default='gpu')

    parser.add_argument('--save_disparity', default=True, type=bool,
                        help='whether store the result of disparity')

    parser.add_argument('--weight', default="", type=str,
                        help="path to the initial weights for the disparity estimation network")

    parser.add_argument('--sampleMode', default='PROBABILITY', choices=sampler_factory.AVAILABLE_SAMPLER,
                        help="choose the sampling heuristic to use")

    parser.add_argument('--numBlocks', default=1, type=int,
                        help="number of CNN portions to train at each iteration")

    parser.add_argument('--fixedID', type=int, nargs='+', default=[0],
                        help="index of the portions of network to train, used only if sampleMode=FIXED")

    parser.add_argument('--reprojectionScale', default=1, type=int)

    parser.add_argument('--SSIMTh', default=0.5, type=float,
                        help='reset network to initial configuration if loss is above this value')

    args = parser.parse_args()
    return args


def main(args):
    # args
    # 加载json配置文件
    with open(args.blockConfig) as json_data:
        train_config = json.load(json_data)

    device = torch.device("cuda" if torch.cuda.is_available() and args.device.lower() == 'gpu' else "cpu")

    # 数据集初始化
    data_transform = transforms.Compose(
        [  # if do data aug?
            # aug
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    dataset = data_reader.dataset(args.path_file, crop_shape=args.crop_shape, augment=False, is_training=False,
                                  transform=None)

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0,
                                             pin_memory=True)

    # dataiter = iter(dataloader)
    # left_img_batch, right_img_batch, gt_image_batch = next(dataiter)

    # 模型参数及模型初始化
    net_args = {}
    net_args['split_layers'] = [None]
    net_args['sequence'] = True
    net_args['train_portion'] = 'BEGIN'
    net_args['bulkhead'] = True if args.mode == 'MAD' else False
    model = madnet(net_args)
    if os.path.exists(args.weight):
        weight_dict = torch.load(args.weight)
        model.load_state_dict(weight_dict, strict=True)
    # model = torch.nn.DataParallel(model.to(device))
    model = model.to(device)
    print(model)



    if args.mode == 'MAD':
        for name, value in model.named_parameters():
            if not value.requires_grad:
                print(name)
            value.requires_grad = False
            # print(name)
            # print(value)

    # 损失函数初始化
    # reconstruction loss between warped right image and original left image
    full_reconstruction_loss = loss_factory.get_reprojection_loss('mean_SSIM_l1', reduced=True)

    # 优化器
    pg = [p for p in model.parameters()]  # if p.requires_grad]
    optimizer = torch.optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5e-5)

    # 学习率
    """
    这里一批数据每次只更新单个模块，所以学习率没有作改变
    """
    # lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    lf = lambda x: args.lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # sampler
    sampler = sampler_factory.get_sampler(args.sampleMode, args.numBlocks, args.fixedID)

    model.train()

    num_actions = len(train_config)
    fetch_counter = [0] * num_actions
    sample_distribution = np.zeros(shape=[num_actions])
    temp_score = np.zeros(shape=[num_actions])
    loss_t_2 = 0
    loss_t_1 = 0
    last_trained_blocks = []
    reset_counter = 0
    step = 0
    epe_accumulator = []
    bad3_accumulator = []
    best_loss = np.inf
    best_epoch = -1

    for epoch in range(args.epochs):
        for left_image, right_image, gt_image in tqdm(dataloader):
            left_image = left_image.to(device)
            right_image = right_image.to(device)
            gt_image = gt_image.to(device)
            inputs = {
                'left': left_image,
                'right': right_image,
                'target': gt_image
            }

            optimizer.zero_grad()
            predictions = model(left_image, right_image)

            full_res_disp = predictions[-1]
            # print("5:{}".format(torch.cuda.memory_allocated(0)))
            full_rc_loss = full_reconstruction_loss(predictions, inputs)
            print("loss:{}".format(full_rc_loss.detach().cpu()))

            # 验证误差
            abs_err = torch.abs(full_res_disp - gt_image)
            valid_map = torch.where(torch.eq(gt_image, 0), torch.zeros_like(gt_image, dtype=torch.float32),
                                    torch.ones_like(gt_image, dtype=torch.float32))
            filtered_error = abs_err * valid_map

            abs_err = torch.sum(filtered_error) / torch.sum(valid_map)
            bad_pixel_abs = torch.where(torch.gt(filtered_error, 3),
                                        torch.ones_like(filtered_error, dtype=torch.float32),
                                        torch.zeros_like(filtered_error, dtype=torch.float32))
            bad_pixel_prec = torch.sum(bad_pixel_abs) / torch.sum(valid_map)

            if args.mode == 'MAD':
                predictions = predictions[:-1]

                # 图片尺度缩放
                inputs_modules = {
                    "left": preprocessing.rescale_image(left_image, (
                        left_image.shape[2] // args.reprojectionScale, left_image.shape[3] // args.reprojectionScale)),
                    "right": preprocessing.rescale_image(right_image, (
                        right_image.shape[2] // args.reprojectionScale,
                        right_image.shape[3] // args.reprojectionScale)),
                    "target": preprocessing.rescale_image(left_image, (
                        gt_image.shape[2] // args.reprojectionScale, gt_image.shape[3] // args.reprojectionScale)),
                }

                distribution = preprocessing.softmax(sample_distribution)
                blocks_to_train = sampler.sample(distribution)
                selected_train_ops = [train_config[i] for i in blocks_to_train]

                for l in blocks_to_train:
                    fetch_counter[l] += 1

                p = predictions[blocks_to_train.item()]
                multiplier = float(left_image.shape[2] // p.shape[2])
                p = preprocessing.resize_to_prediction(p, inputs_modules['left']) * multiplier
                reconstruction_loss = full_reconstruction_loss([p], inputs_modules)

                for name, value in model.named_parameters():
                    title = name.split('.')[0]
                    if title in selected_train_ops:
                        value.requires_grad = True

                reconstruction_loss.requires_grad_(True)
                reconstruction_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                new_loss = reconstruction_loss.detach()
                if step == 0:
                    loss_t_2 = new_loss
                    loss_t_1 = new_loss
                expected_loss = 2 * loss_t_1 - loss_t_2
                gain_loss = expected_loss - new_loss
                sample_distribution = 0.99 * sample_distribution
                for i in last_trained_blocks:
                    sample_distribution[i] += 0.01 * gain_loss
                last_trained_blocks = blocks_to_train
                loss_t_2 = loss_t_1
                loss_t_1 = new_loss

                for value in model.parameters():
                    value.requires_grad = False
                step += 1

                if new_loss > args.SSIMTh:
                    # save model
                    pass

                if new_loss < best_loss:
                    best_loss = new_loss
                    best_epoch = epoch
                    # save model
                    # todo

                print('Done')
                print('=' * 50)



            elif args.mode == 'FULL':
                full_rc_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            epe_accumulator.append(abs_err)
            bad3_accumulator.append(bad_pixel_prec)

            if args.save_disparity:
                out = os.path.join(args.output, "disparity")
                os.makedirs(out, exist_ok=True)
                dispy = full_res_disp.detach().cpu().numpy()
                dispy_to_save = np.clip(dispy[0], 0, MAX_DISP)
                dispy_to_save = (dispy_to_save * 255.0).astype('uint8')
                dispy_to_save = dispy_to_save.transpose(1, 2, 0).astype('uint8')
                cv2.imwrite(os.path.join(args.output, "disparity/disparity_{}.png".format(step)), dispy_to_save)


if __name__ == '__main__':
    args = get_args()
    main(args)
