import datetime
import errno
import math
import os
import random
import shutil
import time
from loguru import logger
import numpy as np
from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
import torch.nn as nn

from utils.logger import setup_logger
from data.data_manager import Init_dataset
from data.data_loader import ImageData
from utils.transforms import TrainTransform, TestTransform
from data.samplers import RandomIdentitySampler
from utils.utils import matplotlib_imshow
from models.networks import Init_model
from utils.loss import CrossEntropyLabelSmooth, TripletLoss
from utils.meters import AverageMeter
from trainers.evaluator import Evaluator, test

import warnings

warnings.filterwarnings("ignore")


class Trainer:
    def __init__(self, args):
        # 初始化日志
        self.args = args
        self.tensorboard = SummaryWriter()
        setup_logger(
            save_dir=self.tensorboard.log_dir,
            distributed_rank=0,
            filename='train_log.txt',  # 'train_log_{time}.txt',
            mode='a'
        )

    def _state_dict(self):
        return {k: getattr(self.args, k) for k, _ in self.args.__dict__.items()}

    def train(self):
        self.before_train()
        try:
            if self.args.evaluate:
                return
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            if self.args.evaluate: return
            self.after_train()

    def before_train(self):
        # 记录日志
        logger.info("===========user config=============")
        # logger.info("Args: {}".format(self.args))
        for k, v in self._state_dict().items():
            logger.info("{:}:{:}".format(k, v))
        logger.info("=============end===================")

        # 初始化设备
        torch.manual_seed(self.args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() and self.args.use_gpu else "cpu")
        if device.type == 'cuda':
            logger.info("Currently using GPU")
            pin_memory = True
            cudnn.benchmark = True
            torch.cuda.manual_seed_all(self.args.seed)
        else:
            pin_memory = False
            logger.info("Currently using cpu")

        # 初始化数据
        logger.info('initializing dataset "{}"'.format(self.args.datatype))
        dataset = Init_dataset(self.args.data_path, self.args.mode)

        logger.info('initializing dataloader "{}"'.format(self.args.datatype))
        nw = min([os.cpu_count(), self.args.batch_size, self.args.num_workers if self.args.batch_size > 1 else 0,
                  8])  # number of workers
        logger.info('Using {} dataloader workers every process'.format(nw))

        self.trainloader = DataLoader(
            ImageData(dataset.train, TrainTransform(self.args.datatype)),
            sampler=RandomIdentitySampler(dataset.train, self.args.num_instances),
            batch_size=self.args.batch_size, num_workers=nw,
            pin_memory=pin_memory, drop_last=True
        )

        self.queryloader = DataLoader(
            ImageData(dataset.query, TestTransform(self.args.datatype)),
            batch_size=self.args.batch_size, num_workers=nw,
            pin_memory=pin_memory
        )

        self.galleryloader = DataLoader(
            ImageData(dataset.gallery, TestTransform(self.args.datatype)),
            batch_size=self.args.batch_size, num_workers=nw,
            pin_memory=pin_memory
        )

        self.queryFliploader = DataLoader(
            ImageData(dataset.query, TestTransform(self.args.datatype, True)),
            batch_size=self.args.batch_size, num_workers=nw,
            pin_memory=pin_memory
        )

        self.galleryFliploader = DataLoader(
            ImageData(dataset.gallery, TestTransform(self.args.datatype, True)),
            batch_size=self.args.batch_size, num_workers=nw,
            pin_memory=pin_memory
        )

        # 将图片写入tensorboard
        logger.info("load train data image to tensorboard...")
        self._train_data2tensorbaord(self.trainloader)

        # 加载模型
        logger.info("initializing model ...")
        model = Init_model(self.args.model_name,
                           [dataset.num_train_pids,
                            self.args.last_stride,
                            self.args.w_ratio,
                            self.args.h_ratio,
                            self.args.global_feature_dim,
                            self.args.part_feature_dim])

        # 如果有预训练模型，则加载预训练模型。默认加载resnet50预训练模型
        if os.path.exists(self.args.weights):
            static_dict = torch.load(self.args.weights)['state_dict']
            static_dict = {k: v for k, v in static_dict.items() if not ('reduction' in k or 'softmax' in k)}
            model.load_state_dict(static_dict, False)
            logger.info('load pretrained model ' + self.args.weights)
        logger.info('model size: {:.5f}M'.format(sum(p.numel() for p in model.parameters()) / 1e6))

        model = nn.DataParallel(model).to(device)
        # 将网络写入tensorboard
        logger.info("load model to tensorboard...")
        if device.type == 'cuda':
            graph_inputs = torch.from_numpy(np.random.rand(1, 3, 224, 224)).type(
                torch.FloatTensor).cuda()
        else:
            graph_inputs = torch.from_numpy(np.random.rand(1, 3, 224, 224)).type(torch.FloatTensor)
        self.tensorboard.add_graph(model, (graph_inputs,))

        reid_evaluate = Evaluator(model)
        if self.args.evaluate:
            reid_evaluate.evaluate(self.queryloader, self.galleryloader,
                                   self.queryFliploader, self.galleryFliploader, re_ranking=self.args.re_ranking,
                                   savefig=self.args.save_fig)
            return

        # 初始化损失函数
        if self.args.CrossEntropy_with_label_smooth:
            xent_criterion = CrossEntropyLabelSmooth(dataset.num_train_pids)
        else:
            xent_criterion = nn.CrossEntropyLoss()

        if self.args.loss == 'triplet':
            embedding_criterion = TripletLoss(self.args.margin)

        # elif self.args.loss == 'lifted':
        #     embedding_criterion = LiftedStructureLoss(hard_mining=True)
        # elif self.args.loss == 'weight':
        #     embedding_criterion = Margin()

        def criterion(triplet_y, softmax_y, labels):
            losses = [embedding_criterion(output, labels)[0] for output in triplet_y] + [xent_criterion(output, labels)
                                                                                         for output in softmax_y]
            loss = sum(losses)
            return loss

        # 优化器
        pg = [p for p in model.parameters() if p.requires_grad]
        if self.args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(pg, lr=self.args.lr, momentum=0.9, weight_decay=5E-5)
        else:
            optimizer = torch.optim.Adam(pg, lr=self.args.lr, weight_decay=1e-3)

        # 学习率
        if self.args.adjust_lr == 'cosine':
            # Scheduler https://arxiv.org/pdf/1812.01187.pdf
            lf = lambda x: ((1 + math.cos(x * math.pi / self.args.max_epoch)) / 2) * (
                    1 - self.args.lrf) + self.args.lrf  # cosine
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        elif self.args.adjust_lr == 'steplr':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(self.args.max_epoch / 4),
                                                                         int(self.args.max_epoch / 2)], 0.1)
        elif self.args.adjust_lr == 'steplr2':
            def adjust_lr(optimizer, ep):
                if ep < 50:
                    lr = 1e-4 * (ep // 5 + 1)
                elif ep < 200:
                    lr = 1e-3
                elif ep < 300:
                    lr = 1e-4
                else:
                    lr = 1e-5
                for p in optimizer.param_groups:
                    p['lr'] = lr

            scheduler = adjust_lr

        self.model = model
        self.device = device
        self.criterion = criterion
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.reid_evaluate = reid_evaluate
        self.best_epoch = 0
        self.best_rank1 = -np.inf

        logger.info("Training start...")
        logger.info("\n{}".format(model))

    def train_in_epoch(self):

        for self.epoch in range(self.args.start_epoch, self.args.max_epoch):
            if self.args.adjust_lr == 'steplr2':
                self.scheduler(self.optimizer, self.epoch + 1)
            logger.info("---> start train epoch{}".format(self.epoch + 1))
            self.model.train()

            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()

            start = time.time()
            data_loader = tqdm(self.trainloader)
            max_iter = len(self.trainloader)
            for i, inputs in enumerate(data_loader):
                data_time.update(time.time() - start)

                self._parse_data(inputs)
                self._forward()
                self.optimizer.zero_grad()
                self._backward()
                self.optimizer.step()

                batch_time.update(time.time() - start)
                losses.update(self.loss.item())

                # 将数据写入tensorboard
                global_step = self.epoch * max_iter + i
                self.tensorboard.add_scalar('loss', self.loss.item(), global_step)
                self.tensorboard.add_scalar('lr', self.optimizer.param_groups[0]['lr'], global_step)

                start = time.time()
                if (i + 1) % self.args.print_freq == 0:
                    logger.info('Epoch: [{}][{}/{}]\t'
                                'Batch Time {:.3f} ({:.3f})\t'
                                'Data Time {:.3f} ({:.3f})\t'
                                'Loss {:.3f} ({:.3f})\t'
                                .format(self.epoch + 1, i + 1, len(data_loader),
                                        batch_time.val, batch_time.mean,
                                        data_time.val, data_time.mean,
                                        losses.val, losses.mean))

            param_group = self.optimizer.param_groups
            logger.info('Epoch: [{}]\tEpoch Time {:.3f} s\tLoss {:.3f}\t'
                        'Lr {:.2e}'
                        .format(self.epoch + 1, batch_time.sum, losses.mean, param_group[0]['lr']))

            if self.args.adjust_lr == 'cosine' or self.args.adjust_lr == 'steplr':
                self.scheduler.step()

            # evaluate
            if self.args.eval_step > 0 and (self.epoch + 1) % self.args.eval_step == 0 or (
                    self.epoch + 1) == self.args.max_epoch:
                if self.args.mode == 'retrieval':
                    rank1 = self.reid_evaluate.evaluate(self.queryloader, self.galleryloader, self.queryFliploader,
                                                        self.galleryFliploader)
                else:
                    rank1 = test(self.model, self.queryloader)

                is_best = rank1 > self.best_rank1
                if is_best:
                    self.best_rank1 = rank1
                    self.best_epoch = self.epoch + 1
                if self.args.use_gpu:
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()
                self._save_checkpoint({'state_dict': state_dict, 'epoch': self.epoch + 1},
                                      is_best=is_best, save_dir=self.tensorboard.log_dir + "/weights",
                                      filename='checkpoint_ep' + str(self.epoch + 1) + '.pth.tar')

    def after_train(self):
        logger.info('Best rank-1 {:.1%}, achived at epoch {}'.format(self.best_rank1, self.best_epoch))

    def _mkdir_if_missing(self, dir_path):
        try:
            os.makedirs(dir_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    def _save_checkpoint(self, state, is_best, save_dir, filename):
        fpath = os.path.join(save_dir, filename)
        self._mkdir_if_missing(save_dir)
        torch.save(state, fpath)
        if is_best:
            shutil.copy(fpath, os.path.join(save_dir, 'model_best.pth.tar'))

    def _parse_data(self, inputs):
        imgs, labels = inputs
        if self.args.random_crop and random.random() > 0.3:
            h, w = imgs.size()[-2:]
            start = int((h - 2 * w) * random.random())
            mask = imgs.new_zeros(imgs.size())
            mask[:, :, start:start + 2 * w, :] = 1
            imgs = imgs * mask
        self.data = imgs.to(self.device)
        self.target = labels.to(self.device)

    def _forward(self):
        triplet_feature, softmax_feature = self.model(self.data)
        self.loss = self.criterion(triplet_feature, softmax_feature, self.target)

    def _backward(self):
        self.loss.backward()

    def _train_data2tensorbaord(self, dataloader):
        dataiter = iter(dataloader)
        images, labels = next(dataiter)
        img_grid = torchvision.utils.make_grid(images, padding=0)
        norm_image, unnorm_image = matplotlib_imshow(img_grid, one_channel=False)
        self.tensorboard.add_image("归一化后的图片", norm_image)
        self.tensorboard.add_figure("原始图片", unnorm_image, global_step=0)
