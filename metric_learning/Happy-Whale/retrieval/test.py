import pandas as pd
import os
import argparse

import torch
from tqdm import tqdm

from dataLoader import *
from models import *
from utils import *


def collate(batch):
    batch_size = len(batch)
    images = []
    names = []
    for b in range(batch_size):
        if batch[b][0] is None:
            continue
        else:
            images.extend(batch[b][0])
            names.append(batch[b][1])
    images = torch.stack(images, 0)
    return images, names


def transform(image, mask):
    raw_iamge = cv2.resize(image, (256, 256))
    raw_mask = cv2.resize(mask, (256, 256))
    raw_mask = raw_mask[:, :, None]
    # raw_iamge = np.concatenate([raw_iamge, raw_mask], 2)

    images = []

    image = raw_iamge.copy()
    image = np.transpose(image, (2, 0, 1))
    image = image.copy().astype(np.float)
    image = torch.from_numpy(image).div(255).float()
    images.append(image)

    image = raw_iamge.copy()
    image = np.fliplr(image)
    image = np.transpose(image, (2, 0, 1))
    image = image.copy().astype(np.float)
    image = torch.from_numpy(image).div(255).float()
    images.append(image)

    return images


def run(args):
    assert os.path.exists(args.test_dir)
    assert os.path.exists(args.csv_path)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    # 数据
    df = pd.read_csv(args.csv_path)
    file_names = df['image']
    dst_test = WhaleTestDataset({"name": file_names, "img_path": args.test_dir}, mode='test', transform=transform)
    dataloader_test = DataLoader(dst_test, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate)

    dataiter = iter(dataloader_test)
    images, names = next(dataiter)

    # 标签索引
    json_file = open('./data/class_indices.json', "r")
    class_indict = json.load(json_file)
    id_label = {v: k for k, v in class_indict.items()}
    id_label[args.num_classes] = 'new_whale'

    # 模型
    model = model_whale(num_classes=args.num_classes * 2, inchannels=3, model_name=args.model_name).to(device)

    result_dir = './result/{}_{}'.format(args.model_name, args.fold_index)
    check_point = result_dir + '/checkpoint'

    npy_dir = result_dir + '/out_{}'.format(args.checkpoint_start)
    os.makedirs(npy_dir, exist_ok=True)

    if not args.checkpoint_start == 0:
        model.load_pretrain(os.path.join(check_point, '%08d_model_path' % args.checkpoint_start), skip=[])
        ckp = torch.load(os.path.join(check_point, '%08d_optimizer.pth' % args.checkpoint_start))
        best_t = ckp['best_t']
        print('best_t:', best_t)
    # best_t=0.5

    labelstrs = []
    allnames = []
    with torch.no_grad():
        model.eval()
        for data in tqdm(dataloader_test):
            images, names = data
            images = images.to(device)
            _, _, outs = model(images)
            outs = torch.sigmoid(outs)
            outs_zero = (outs[::2, :args.num_classes] + outs[1::2, args.num_classes:]) / 2
            outs = outs_zero
            for out, name in zip(outs, names):
                out = torch.cat([out, torch.ones(1).cuda() * best_t], 0)
                out = out.data.cpu().numpy()
                np.save(os.path.join(npy_dir, '{}.npy'.format(name)), out)
                top5 = out.argsort()[-5:][::-1]
                str_top5 = ''
                for t in top5:
                    str_top5 += '{} '.format(id_label[t])
                str_top5 = str_top5[:-1]
                allnames.append(name)
                labelstrs.append(str_top5)
    pd.DataFrame({'Image': allnames, 'Id': labelstrs}).to_csv(
        'test_{}_sub_fold{}.csv'.format(args.model_name, args.fold_index), index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', default=r'E:\dataset\happy_whale_cropped\test_images', type=str,
                        help='path of test image data')
    parser.add_argument('--csv_path', default=r'E:\dataset\happy_whale_cropped\test.csv', type=str,
                        help='path of test csv file')
    parser.add_argument('--checkpoint_start', default=0)
    parser.add_argument('--fold_index', default=1, type=int)
    parser.add_argument('--model_name', default='senet154', type=str)
    parser.add_argument('--num_classes', default=15587, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--use_cuda', default=True)

    args = parser.parse_args()
    run(args)
