import argparse
import datetime
import os
import sys
from PIL import Image
import numpy as np

from models.networks import BFE
from trainers.re_ranking import re_ranking as re_ranking_func

import torch
from torch.utils.data import DataLoader
from utils.transforms import TestTransform
from torchvision import transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset

import warnings

warnings.filterwarnings("ignore")


def run(opt):
    model = init_model(opt.weights, opt.use_gpu)
    galleryloader, galleryloaderFlip = init_gallery(
        opt.gallery_path, opt)
    queryloader, queryloaderFlip = init_query(opt.query_path, opt)
    infer(model, galleryloader, galleryloaderFlip, queryloader, queryloaderFlip, opt)


def infer(model, galleryloader, galleryloaderFlip, queryloader, queryloaderFlip, opt):
    model.eval()
    start = datetime.datetime.now()
    qf = []
    gf = []
    scores = []
    for inputs0, inputs1 in zip(queryloader, queryloaderFlip):
        # inputs, pids = parse_data(inputs0, opt)
        inputs = parse_data(inputs0, opt)
        feature0 = forward(model, inputs)
        if opt.eval_flip:
            # inputs, pids = parse_data(inputs1, opt)
            inputs = parse_data(inputs1, opt)
            feature1 = forward(model, inputs)
            qf.append((feature0 + feature1) / 2.0)
        else:
            qf.append(feature0)
    qf = torch.cat(qf, 0)
    if len(qf.shape) == 1:
        qf = qf[None, :]
    print("Extracted features for query set: {} x {}".format(qf.size(0), qf.size(1)))

    for inputs0, inputs1 in zip(galleryloader, galleryloaderFlip):
        # inputs, pids = parse_data(inputs0, opt)
        inputs = parse_data(inputs0, opt)
        feature0 = forward(model, inputs)
        if opt.eval_flip:
            # inputs, pids = parse_data(inputs1, opt)
            inputs = parse_data(inputs1, opt)
            feature1 = forward(model, inputs)
            gf.append((feature0 + feature1) / 2.0)
        else:
            gf.append(feature0)
    gf = torch.cat(gf, 0)
    if len(gf.shape) == 1:
        gf = gf[None, :]
    print("Extracted features for gallery set: {} x {}".format(gf.size(0), gf.size(1)))

    print("Computing distance matrix")

    m, n = qf.size(0), gf.size(0)
    q_g_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    q_g_dist.addmm_(1, -2, qf, gf.t())

    if opt.re_ranking:
        q_q_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m) + \
                   torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m).t()
        q_q_dist.addmm_(1, -2, qf, qf.t())

        g_g_dist = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n) + \
                   torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n).t()
        g_g_dist.addmm_(1, -2, gf, gf.t())

        q_g_dist = q_g_dist.numpy()
        q_g_dist[q_g_dist < 0] = 0
        q_g_dist = np.sqrt(q_g_dist)

        q_q_dist = q_q_dist.numpy()
        q_q_dist[q_q_dist < 0] = 0
        q_q_dist = np.sqrt(q_q_dist)

        g_g_dist = g_g_dist.numpy()
        g_g_dist[g_g_dist < 0] = 0
        g_g_dist = np.sqrt(g_g_dist)

        distmat = torch.Tensor(re_ranking_func(q_g_dist, q_q_dist, g_g_dist))
    else:
        distmat = q_g_dist

    # print(distmat)
    num_q, num_g = distmat.size()
    _, indices = torch.sort(distmat, dim=1)

    # cat = gids[indices][:, :opt.topk].tolist()
    if opt.topk == 1:
        score = (1 - F.normalize(_, p=2, dim=1))[:, 0].tolist()
    else:
        score = (1 - F.normalize(_[:, :opt.topk], p=2, dim=1)).tolist()
    # result.extend(cat)
    scores.extend(score)
    end = datetime.datetime.now()
    t = (end - start).seconds
    print('Match feature Time Elapsed: {}s'.format(t))
    print(score)
    print(indices)
    # scores:排序后的分数
    # indices:每个query与gallery中的图片按相似度排序，indices就是gallery_list图片的索引。
    return scores, indices


def parse_data(input, opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() and opt.use_gpu else "cpu")
    imgs = input
    return imgs.to(device)

    # imgs, pids = input
    # return imgs.to(device), pids.to(device)


def forward(model, input):
    with torch.no_grad():
        feature = model(input)
    return feature.cpu()


def init_model(weights, use_cuda=True):
    # print("=========initializing model===========")
    assert os.path.exists(weights), "ERROR model weighs: {} does not exists".format(weights)
    device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")

    static_dict = torch.load(weights, map_location=device)['state_dict']
    for key, value in static_dict.items():
        if key == 'global_softmax.weight':
            global_feature_dim = value.shape[1]
            num_classes = value.shape[0]
        elif key == 'part_softmax.weight':
            part_feature_dim = value.shape[1]
            num_classes = value.shape[0]

    model = BFE(
        num_classes=num_classes,
        global_feature_dim=global_feature_dim,
        part_feature_dim=part_feature_dim
    )
    model.load_state_dict(static_dict, False)
    model.to(device)
    model.eval()
    return model


def init_query(source, opt):
    # print("=========initializing query===========")
    assert os.path.exists(source)
    query_list = []
    if os.path.isfile(source):
        query_list.append(source)
    elif os.path.isdir(source):
        query_list.extend(os.path.join(source, i) for i in os.listdir(source))

    queryloader = DataLoader(
        ImageData(dataset=query_list, transform=Transform(flip=False, image_size=opt.image_size)),
        batch_size=opt.batch_size, num_workers=opt.num_worker,
        pin_memory=True if torch.cuda.is_available() and opt.use_gpu else False,
        shuffle=False
    )
    queryloaderFlip = DataLoader(
        ImageData(dataset=query_list, transform=Transform(flip=True, image_size=opt.image_size)),
        batch_size=opt.batch_size, num_workers=opt.num_worker,
        pin_memory=True if torch.cuda.is_available() and opt.use_gpu else False,
        shuffle=False
    )

    return queryloader, queryloaderFlip


def init_gallery(source, opt):
    # print("=========initializing gallery===========")
    assert os.path.exists(source)
    gallery_list = []
    if os.path.isfile(source):
        gallery_list.append(source)
    elif os.path.isdir(source):
        gallery_list.extend(os.path.join(source, i) for i in os.listdir(source))

    galleryloader = DataLoader(
        ImageData(dataset=gallery_list, transform=Transform(flip=False, image_size=opt.image_size)),
        batch_size=opt.batch_size, num_workers=opt.num_worker,
        pin_memory=True if torch.cuda.is_available() and opt.use_gpu else False,
        shuffle=False
    )
    galleryloaderFlip = DataLoader(
        ImageData(dataset=gallery_list, transform=Transform(flip=True, image_size=opt.image_size)),
        batch_size=opt.batch_size, num_workers=opt.num_worker,
        pin_memory=True if torch.cuda.is_available() and opt.use_gpu else False,
        shuffle=False
    )

    return galleryloader, galleryloaderFlip


class Transform(object):
    def __init__(self, flip=False, image_size=224):
        self.flip = flip
        self.image_size = image_size

    def __call__(self, x=None):
        def pad_shorter(x):
            h, w = x.size[-2:]
            s = max(h, w)
            new_im = Image.new("RGB", (s, s))
            new_im.paste(x, ((s - h) // 2, (s - w) // 2))
            return new_im

        x = pad_shorter(x)
        x = T.Resize((self.image_size, self.image_size))(x)

        if self.flip:
            x = T.functional.hflip(x)
        x = T.ToTensor()(x)
        x = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])(x)
        return x


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageData(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, item):
        img = self.dataset[item]
        img = read_image(img)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.dataset)


def result():
    pass


def parser_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query-path', type=str, default='./data/query')
    parser.add_argument('--gallery-path', type=str, default='./data/gallery')
    parser.add_argument('--datatype', type=str, default='button')
    parser.add_argument('--mode-type', type=str, default='retrieval')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--model-name', type=str, default='bfe')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-worker', type=int, default=1)
    parser.add_argument('--weights', type=str, default='./model_best.pth.tar')
    parser.add_argument('--use-gpu', default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--re-ranking', type=bool, default=False)
    parser.add_argument('--savefig', type=bool, default=False)
    parser.add_argument('--eval-flip', type=bool, default=False)
    parser.add_argument('--topk', type=int, default=1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser_arg()
    run(opt=args)
