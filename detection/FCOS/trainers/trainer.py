import os
import time
import warnings

import numpy as np
import random
from argparse import Namespace

import torchvision
from loguru import logger

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

from utils.logger import setup_logger
from models.fcos import FCOSDetector
from data.voc import VOCDetection
from data.data_augment import Transforms
from utils.utils import matplotlib_imshow
from .eval_voc import sort_by_score, eval_ap_2d


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
            self.train_in_epoch()
        except Exception:
            raise
        finally:
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
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.use_gpu else "cpu")
        if self.device.type == 'cuda':
            logger.info("Currently using GPU")
            pin_memory = True
            cudnn.benchmark = True
            cudnn.deterministic = True
            torch.cuda.manual_seed(self.args.seed)
            torch.cuda.manual_seed_all(self.args.seed)
        else:
            pin_memory = False
            logger.info("Currently using cpu")

        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

        # model
        logger.info('init fcos model ...')
        self.model = FCOSDetector(mode='training', cfg='./models/model.yaml')
        logger.info('model to dataparallel ...')
        self.model = torch.nn.DataParallel(self.model).to(device=self.device)

        # 将网络写入tensorboard
        logger.info("load model to tensorboard...")
        if self.device.type == 'cuda':
            graph_inputs = torch.from_numpy(np.random.rand(1, 3, 224, 224)).type(
                torch.FloatTensor).cuda()
        else:
            graph_inputs = torch.from_numpy(np.random.rand(1, 3, 224, 224)).type(torch.FloatTensor)
        self.eval_model = FCOSDetector(mode='inference', cfg='./models/model.yaml')
        self.eval_model = torch.nn.DataParallel(self.eval_model).to(device=self.device)
        self.tensorboard.add_graph(self.eval_model, (graph_inputs,))

        if self.args.pretrain_weights is not None:
            assert os.path.exists(self.args.pretrain_weights), "ERROR pretrain weight {} does not exists!".format(
                self.args.pretrain_weights)
            self.model.load_state_dict(torch.load(self.args.pretrain_weights, device=self.device), strict=False)
            logger.info('load pretrain weights from {}'.format(self.args.pretrain_weights))

        # dataset
        logger.info('init dataset from {}'.format(self.args.data_type))
        if self.args.data_type == 'voc':
            self.train_dataset = VOCDetection(data_dir=self.args.data_path,
                                              image_sets=[("2012", "trainval")],
                                              img_size=(416, 416),
                                              preproc=Transforms(),
                                              is_train=True)

            self.eval_dataset = VOCDetection(data_dir=self.args.data_path,
                                             image_sets=[("2012", "val")],
                                             img_size=(416, 416),
                                             preproc=None,
                                             is_train=False)

        elif self.args.data_type == 'coco':
            self.dataset = 1
        else:
            self.dataset = 1

        # dataloader
        logger.info('init train loader ...')
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=self.args.batch_size,
                                                        shuffle=True,
                                                        collate_fn=self.train_dataset.collate_fn,
                                                        num_workers=self.args.workers, worker_init_fn=np.random.seed(0))

        logger.info('init val loader ...')
        self.eval_loader = torch.utils.data.DataLoader(self.eval_dataset,
                                                       batch_size=self.args.batch_size,
                                                       shuffle=False,
                                                       collate_fn=self.eval_dataset.collate_fn,
                                                       num_workers=self.args.workers, worker_init_fn=np.random.seed(0))

        # 将图片写入tensorboard
        logger.info("load train data image to tensorboard...")
        self._train_data2tensorbaord(self.train_loader)

        self.steps_per_epoch = len(self.train_dataset) // self.args.batch_size
        logger.info('len of train dataset : {}'.format(len(self.train_dataset)))
        logger.info('steps per epoch : {}'.format(self.steps_per_epoch))

        self.total_steps = self.steps_per_epoch * self.args.epochs
        logger.info('total steps : {}'.format(self.total_steps))

        if self.args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr_init, momentum=0.9,
                                             weight_decay=1e-4)
        elif self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr_init, weight_decay=1e-4)

        self.best_map = 0
        self.best_epoch = 0
        os.makedirs(self.args.save_path, exist_ok=True)

    def train_in_epoch(self):
        logger.info('start train ...')
        GLOBAL_STEPS = 1
        self.model.train()
        for epoch in range(self.args.epochs):
            self.epoch = epoch
            for epoch_step, data in enumerate(self.train_loader):
                batch_imgs, batch_boxes, batch_classes = data
                batch_imgs = batch_imgs.to(self.device)
                batch_boxes = batch_boxes.to(self.device)
                batch_classes = batch_classes.to(self.device)

                if GLOBAL_STEPS < self.args.warmup_steps:
                    lr = float(GLOBAL_STEPS / self.args.warmup_steps * self.args.lr_init)
                    for param in self.optimizer.param_groups:
                        param['lr'] = lr
                if GLOBAL_STEPS == 20001:
                    lr = self.args.lr_init * 0.1
                    for param in self.optimizer.param_groups:
                        param['lr'] = lr
                if GLOBAL_STEPS == 27001:
                    lr = self.args.lr_init * 0.01
                    for param in self.optimizer.param_groups:
                        param['lr'] = lr
                start_time = time.time()

                self.optimizer.zero_grad()
                losses = self.model([batch_imgs, batch_boxes, batch_classes])
                loss = losses[-1]
                loss.mean().backward()
                self.optimizer.step()

                end_time = time.time()
                cost_time = int((end_time - start_time) * 1000)

                # 将数据写入tensorboard
                self.tensorboard.add_scalar('total loss', loss.item(), GLOBAL_STEPS)
                self.tensorboard.add_scalar('cls loss', losses[0].mean(), GLOBAL_STEPS)
                self.tensorboard.add_scalar('cnt loss', losses[1].mean(), GLOBAL_STEPS)
                self.tensorboard.add_scalar('reg loss', losses[2].mean(), GLOBAL_STEPS)
                self.tensorboard.add_scalar('lr', self.optimizer.param_groups[0]['lr'], GLOBAL_STEPS)

                logger.info(
                    "global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e total_loss:%.4f" % \
                    (GLOBAL_STEPS, epoch + 1, epoch_step + 1, self.steps_per_epoch, losses[0].mean(), losses[1].mean(),
                     losses[2].mean(), cost_time, lr, loss.mean()))

                GLOBAL_STEPS += 1

            # 保存模型
            torch.save(self.model.state_dict(),
                       os.path.join(self.args.save_path, "model_{}.pth".format(self.epoch + 1)))
            # eval
            # if epoch % self.args.per_eval == 0 or epoch == self.args.epochs - 1:
            #     logger.info('start eval ...')
            #     self.eval_model.load_state_dict(self.model.state_dict())
            #     self.eval_model.eval()
            #     map = self.eval()
            #     if map > self.best_map:
            #         self.best_map = map
            #         self.best_epoch = epoch
            #         torch.save(self.model.state_dict(), os.path.join(self.args.save_path, "best.pth"))

    def after_train(self):
        logger.info("over!!!")
        logger.info("best map is {} in epoch:{}".format(self.best_map, self.best_epoch))


    def _train_data2tensorbaord(self, dataloader):
        dataiter = iter(dataloader)
        images, _, _ = next(dataiter)
        img_grid = torchvision.utils.make_grid(images, padding=0)
        norm_image, unnorm_image = matplotlib_imshow(img_grid, one_channel=False)
        self.tensorboard.add_image("归一化后的图片", norm_image)
        self.tensorboard.add_figure("原始图片", unnorm_image, global_step=0)

    def eval(self):
        gt_boxes = []
        gt_classes = []
        pred_boxes = []
        pred_classes = []
        pred_scores = []
        num = 0
        for img, boxes, classes in self.eval_loader:
            with torch.no_grad():
                out = self.eval_model(img.to(self.device))
                pred_boxes.append(out[2][0].cpu().numpy())
                pred_classes.append(out[1][0].cpu().numpy())
                pred_scores.append(out[0][0].cpu().numpy())
            gt_boxes.append(boxes[0].numpy())
            gt_classes.append(classes[0].numpy())
            num += 1
            logger.info(str(num), end='\r')

        pred_boxes, pred_classes, pred_scores = sort_by_score(pred_boxes, pred_classes, pred_scores)
        all_AP = eval_ap_2d(gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores, 0.5,
                            len(self.eval_dataset.classes))
        logger.info("all classes AP=====>\n")
        for key, value in all_AP.items():
            logger.info('ap for {} is {}'.format(self.eval_dataset.id2name[int(key)], value))
        mAP = 0.
        for class_id, class_mAP in all_AP.items():
            mAP += float(class_mAP)
        mAP /= (len(self.eval_dataset.classes) - 1)
        logger.info("mAP=====>%.3f\n" % mAP)
        return mAP
