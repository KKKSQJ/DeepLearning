import os
import time
from timeit import default_timer as timer
import cv2
import numpy as np
import datetime
import argparse
import torch
from torch.nn.parallel.data_parallel import data_parallel

from dataLoader import *
from utils import *
from models import *
import warnings
warnings.filterwarnings("ignore")


def train(args):
    # log
    result_dir = './result/{}_{}'.format(args.model_name, args.fold_index)
    image_dir = result_dir + '/image'
    checkpoint_dir = result_dir + '/checkpoint'
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    log = Logger()
    log.open(os.path.join(result_dir, 'log_train.txt'), mode='a')
    log.write(' start_time :{} \n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    log.write('===========user config=============\n')
    # logger.info("Args: {}".format(self.args))
    for k, v in args.__dict__.items():
        log.write("{:}:{:}\n".format(k, v))
    log.write("=============end===================\n")

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    # 数据
    # get_k_fold(csv_dir=args.csv_dir, k=5)
    class_dict = get_classes_id_json(csv_file=args.csv_dir, save_dir='./data')
    train_data = get_input_data(csv_dir='data/train_split_{}.csv'.format(args.fold_index), img_dir=args.image_dir,
                                mask_dir=args.mask_dir)
    val_data = get_input_data(csv_dir='data/val_split_{}.csv'.format(args.fold_index), img_dir=args.image_dir,
                              mask_dir=args.mask_dir)
    dst_train = WhaleDataset(train_data, class_id_label_file=class_dict, mode='train', transform=transform_train,
                             min_num_classes=args.min_num_class)
    train_dataloader = DataLoader(dst_train, shuffle=True, drop_last=True, batch_size=args.batch_size,
                                  num_workers=args.num_workers, collate_fn=train_collate)

    # dataiter = iter(train_dataloader)
    # images, labels = next(dataiter)

    # print(dst_train.__len__())
    num_data = len(train_data['name'])

    dst_val = WhaleTestDataset(val_data, class_id_label_file=class_dict, mode='valid', transform=transform_valid)
    val_dataloader = DataLoader(dst_val, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers,
                                collate_fn=valid_collate)

    # dataiter = iter(val_dataloader)
    # images, labels,names = next(dataiter)

    # 模型
    model = model_whale(num_classes=args.num_classes*2, inchannels=3, model_name=args.model_name).to(device)
    i = 0
    iter_smooth = 50
    iter_valid = 200
    iter_save = 200
    epoch = 0
    if args.freeze:
        model.freeze()

    # 优化器
    if args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0002)
    elif args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0002)
    elif args.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.002)


    # checkpoint
    skips = []
    if not args.checkpoint_start == 0:
        log.write('  start from {}, l_rate ={} \n'.format(args.checkpoint_start, args.lr))
        log.write(
            '  freeze={}, batch_size={}, min_num_class={} \n'.format(args.freeze, args.batch_size, args.min_num_class))
        model.load_pretrain(os.path.join(checkpoint_dir, '%08d_model.pth' % args.checkpoint_start), skip=skips)
        ckp = torch.load(os.path.join(checkpoint_dir, '%08d_optimizer.pth' % args.checkpoint_start))
        optimizer.load_state_dict(ckp['optimizer'])
        adjust_learning_rate(optimizer, args.lr)
        i = args.checkpoint_start
        epoch = ckp['epoch']
    log.write(
        ' rate     iter   epoch  | valid   top@1    top@5    map@5  | '
        'train    top@1    top@5    map@5 |'
        ' batch    top@1    top@5    map@5 |  time          \n')
    log.write(
        '----------------------------------------------------------------------\n')
    start = timer()

    start_epoch = epoch
    best_t = 0
    train_loss = 0.0
    valid_loss = 0.0
    top1, top5, map5 = 0, 0, 0
    top1_train, top5_train, map5_train = 0, 0, 0
    top1_batch, top5_batch, map5_batch = 0, 0, 0

    batch_loss = 0.0
    train_loss_sum = 0
    train_top1_sum = 0
    train_map5_sum = 0
    sum = 0
    while i < 10000000:
        for data in train_dataloader:
            epoch = start_epoch + (i - args.checkpoint_start) * 4 * args.batch_size / num_data
            if i % iter_valid == 0:
                valid_loss, top1, top5, map5, best_t = \
                    eval(model, val_dataloader, device)
                print('\r', end='', flush=True)

                log.write(
                    '%0.5f %5.2f k %5.2f  |'
                    ' %0.3f    %0.3f    %0.3f    %0.4f    %0.4f | %0.3f    %0.3f    %0.3f | %0.3f     %0.3f    %0.3f | %s \n' % ( \
                        args.lr, i / 1000, epoch,
                        valid_loss, top1, top5, map5, best_t,
                        train_loss, top1_train, map5_train,
                        batch_loss, top1_batch, map5_batch,
                        time_to_str((timer() - start) / 60)))
                time.sleep(0.01)

            if i % iter_save == 0 and not i == args.checkpoint_start:
                torch.save(model.state_dict(), result_dir + '/checkpoint/%08d_model.pth' % (i))
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter': i,
                    'epoch': epoch,
                    'best_t': best_t,
                }, result_dir + '/checkpoint/%08d_optimizer.pth' % (i))

            model.train()
            model.mode = 'train'
            images, labels = data
            images = images.to(device)
            labels = labels.to(device).long()
            global_feat, local_feat, results = data_parallel(model, images)
            model.getLoss(global_feat, local_feat, results, labels)
            batch_loss = model.loss
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
            optimizer.step()
            results = torch.cat([torch.sigmoid(results), torch.ones_like(results[:, :1]).float().cuda() * 0.5], 1)
            results = torch.sigmoid(results)
            top1_batch = accuracy(results, labels, topk=(1,))[0]
            map5_batch = mapk(labels, results, k=5)

            batch_loss = batch_loss.data.cpu().numpy()
            sum += 1
            train_loss_sum += batch_loss
            train_top1_sum += top1_batch
            train_map5_sum += map5_batch
            if (i + 1) % iter_smooth == 0:
                train_loss = train_loss_sum / sum
                top1_train = train_top1_sum / sum
                map5_train = train_map5_sum / sum
                train_loss_sum = 0
                train_top1_sum = 0
                train_map5_sum = 0
                sum = 0

            # print(
            #     '\r%0.5f %5.2f k %5.2f  | %0.3f    %0.3f    %0.3f    %0.4f    %0.4f | %0.3f    %0.3f    %0.3f | %0.3f     %0.3f    %0.3f | %s  %d %d' % ( \
            #         args.lr, i / 1000, epoch,
            #         valid_loss, top1, top5, map5, best_t,
            #         train_loss, top1_train, map5_train,
            #         batch_loss, top1_batch, map5_batch,
            #         time_to_str((timer() - start) / 60), args.checkpoint_start, i)
            #     , end='', flush=True)
            log.write(
                '\r%0.5f %5.2f k %5.2f  | %0.3f    %0.3f    %0.3f    %0.4f    %0.4f | %0.3f    %0.3f    %0.3f | %0.3f     %0.3f    %0.3f | %s  %d %d' % ( \
                    args.lr, i / 1000, epoch,
                    valid_loss, top1, top5, map5, best_t,
                    train_loss, top1_train, map5_train,
                    batch_loss, top1_batch, map5_batch,
                    time_to_str((timer() - start) / 60), args.checkpoint_start, i)
                )
            i += 1
        pass


def eval(model, dataLoader_valid, device):
    with torch.no_grad():
        model.eval()
        model.mode = 'valid'
        valid_loss, index_valid = 0, 0
        all_results = []
        all_labels = []
        for valid_data in dataLoader_valid:
            images, labels, names = valid_data
            images = images.to(device)
            labels = labels.to(device).long()
            feature, local_feat, results = data_parallel(model, images)
            # 这里之所以取[::2]前一半的数据，是因为images是两倍的原始数据，后一半做了翻转操作
            model.getLoss(feature[::2], local_feat[::2], results[::2], labels)
            results = torch.sigmoid(results)
            results_zeros = (results[::2, :15587] + results[1::2, 15587:]) / 2
            # 前一半是原图，后一半是翻转后的图
            # results_zeros = (results[::2, :] + results[1::2, :]) / 2
            all_results.append(results_zeros)
            all_labels.append(labels)
            b = len(labels)
            valid_loss += model.loss.data.cpu().numpy() * b
            index_valid += b
        all_results = torch.cat(all_results, 0)
        all_labels = torch.cat(all_labels, 0)
        map5s, top1s, top5s = [], [], []
        if 1:
            ts = np.linspace(0.1, 0.9, 9)
            for t in ts:
                results_t = torch.cat([all_results, torch.ones_like(all_results[:, :1]).float().cuda() * t], 1)
                # all_labels[all_labels == 5004 * 2] = 5004
                top1_, top5_ = accuracy(results_t, all_labels)
                map5_ = mapk(all_labels, results_t, k=5)
                map5s.append(map5_)
                top1s.append(top1_)
                top5s.append(top5_)
            map5 = max(map5s)
            i_max = map5s.index(map5)
            top1 = top1s[i_max]
            top5 = top5s[i_max]
            best_t = ts[i_max]

        valid_loss /= index_valid
        return valid_loss, top1, top5, map5, best_t


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', default=True)
    parser.add_argument('--model_name', default='senet154', type=str, help='model name')
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--csv_dir', default=r'E:\dataset\happy_whale_cropped\train.csv', type=str,
                        help="path of csv dir")
    parser.add_argument('--image_dir', default=r'E:\dataset\happy_whale_cropped\train_images', type=str,
                        help="path of image dir")
    parser.add_argument('--optimizer',default='adam')
    parser.add_argument('--mask_dir', default='', type=str, help="path of mask dir")
    parser.add_argument('--checkpoint_start', default=0, type=int)
    parser.add_argument('--fold_index', default=1, type=int, help='k fold index')
    parser.add_argument('--num_classes', default=15587, type=int, help='num of classes')
    parser.add_argument('--min_num_class', default=10, type=int, help='first is 10,then is 0')
    parser.add_argument('--batch_size', default=6, type=int, help='batch size') #36
    parser.add_argument('--num_workers', default=0, type=int, help='num of workers')#18
    parser.add_argument('--freeze', default=False, type=str)

    args = parser.parse_args()

    train(args)
