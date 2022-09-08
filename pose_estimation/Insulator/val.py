from pathlib import Path
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models import HighResolution as hrnet
from dataset import Keypoint
from dataset import kp_transforms as transforms
from utils import select_device, increment_path


def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data', help='dataset path')
    parser.add_argument('--weights', type=str, default='best.pth', help='weights path')

    opt = parser.parse_args()

    return opt


@torch.no_grad()
def run(
        data,
        weights=None,
        batch_size=3,
        base_channel=32,
        num_joint=1,
        imgsz=448,
        conf_thresh=0.65,
        dis_thresh=2,
        device='',
        workers=0,
        save_txt=True,
        project=ROOT / 'runs/val',
        name='exp',
        exist_ok=False,
):
    # device
    device = select_device(device, batch_size=batch_size)

    # save dir
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    # load model
    model = hrnet(base_channel=base_channel, num_joint=num_joint)
    # load weights
    if os.path.exists(weights):
        ckpt = torch.load(weights, map_location='cpu')
        ckpt = ckpt['state_dict']
        model.load_state_dict(ckpt, strict=True)
    else:
        raise RuntimeError(f"weights {weights} does not exists!!!")

    model.to(device)

    # data
    assert os.path.exists(data), f"data path {data} does not exist!"
    img_path = os.path.join(data, "images")
    anno_path = os.path.join(data, "annos")
    assert os.path.exists(img_path), f"data path {img_path} does not exist!"
    assert os.path.exists(anno_path), f"data path {anno_path} does not exist!"

    data_transforms = transforms.Compose(
        [
            transforms.AffineTransform(scale=(1.0, 1.0), fixed_size=(imgsz, imgsz)),
            transforms.KeypointToHeatMap(heatmap_hw=(imgsz // 4, imgsz // 4), gaussian_sigma=2,
                                         keypoints_nums=num_joint),
            transforms.ToTensor(),
            transforms.Normalize([0.616, 0.231, 0.393], [0.312, 0.288, 0.250])
            # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

    data_set = Keypoint(img_path=img_path, anno_path=anno_path, transforms=data_transforms)
    dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True,
                            drop_last=False, collate_fn=data_set.collate_fn)

    model.eval()
    cuda = device.type != 'cpu'

    # val

    # s = ('%20s' + '%11s' * 3) % ('Image', 'P', 'R', 'mAP')
    # pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    # conf_thres
    conf_thresh = torch.linspace(0.5, 0.99, 50, device=device) if conf_thresh is None else torch.ones(1,
                                                                                                      device=device) * conf_thresh
    nconf = conf_thresh.numel()
    disv = torch.linspace(0, 10, 10, device=device) if dis_thresh is None else torch.ones(1, device=device) * dis_thresh
    ndis = disv.numel()

    if save_txt:
        f = open(str(save_dir) + "/result.txt", 'w')

    for i in range(nconf):
        thresh = conf_thresh[i].item()
        total_images = 0
        total_gts = 0
        total_precs = 0
        stats = []
        s = ('%20s' + '%11s' * 5) % ('Image', 'P', 'R', 'f1', 'mAP', 'thresh')
        pbar = tqdm(dataloader, desc=s, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        for batch_i, (images, targets) in enumerate(pbar):
            # data_time.update(time.time() - end)
            total_images += images.shape[0]
            imgs = torch.stack([image.to(device) for image in images])
            preds = model(imgs)

            out, _ = get_final_preds(preds, img_size=(imgsz, imgsz), thresh=thresh, max_kp=50)

            # 遍历每一张图片的预测结果
            for si, pred in enumerate(out):
                target = targets[si]
                labels_kp = target['labels'][:, None]
                gt_kp = target['keypoints']
                gts = torch.from_numpy(np.concatenate([gt_kp, labels_kp], axis=1)).to(device)
                nl, npr = gts.shape[0], pred.shape[0]
                total_gts += nl
                total_precs += npr
                correct = torch.zeros(npr, ndis, dtype=torch.bool, device=device)

                if npr == 0:
                    if nl:
                        stats.append((correct, *torch.zeros(3, 0, device=device)))
                    continue

                predn = pred.clone()

                if nl:
                    # correct:[n,m] n:预测的点的个数 m：阈值个数 表示每个点在不同阈值在是否为tp。
                    # correct矩阵中为true的样本即为正样本
                    correct = process_batch(predn, gts, disv)
                stats.append((correct, pred[:, 2], pred[:, 3], gts[:, 2]))  # (correct, conf, pcls, tcls)

        # 将所有图片的预测结果整合到一起
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]

        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir)
            # ap_first, ap = ap[:, 0], ap.mean(1)  # 第一个阈值的ap，所有阈值的平均ap
            # mp, mr, map_first, map = p.mean(), r.mean(), ap_first.mean(), ap.mean()
            # nt = np.bincount(stats[3].astype(int), minlength=nc)  # number of targets per class
        # else:
        #     nt = torch.zeros(1)

        if save_txt:
            # 置信度 Images数量 GT数量 预测点的数量 TP FP P R F1 mAP 距离阈值
            f.write(
                "confidence: {}\nImages: {}\nGTS: {}\nPrecisions: {}\nTP: {}\nFP: {}\nP: {}\nR:{}\nF1: {}\nAP: {}\ndistance thresh: {}\n\n".format(
                    thresh, total_images,
                    total_gts,
                    total_precs, tp, fp,
                    p, r, f1, ap, disv.cpu().numpy()))

        pf = '%5s' + '%11i' * 1 + '%11.3g' * 5  # print format
        # pbar.set_description(pf % ('keypoints', total_images, float(p[3]), float(r[3]), float(f1[3]), float(ap[0][3]), thresh))
        print('\n')
        # print(pf % ('keypoints', total_images, float(p[3]), float(r[3]), float(f1[3]), float(ap[0][3]), thresh))
        print(pf % ('keypoints', total_images, float(p[0]), float(r[0]), float(f1[0]), float(ap[0][0]), thresh))
        pbar.close()

    f.close()



def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=(), eps=1e-16):
    # 按照置信度进行排序
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # 找到关键点类别数量
    # unique_classes:类别列表，如[0,1,2...]  nt:GT数量
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # 类别数量

    # 创建p-r曲线，并且为每一个类别计算ap
    # 创建x轴（recall），y轴(precision)
    px, py = np.linspace(0, 1, 1000), []
    # ap:n个类别在m个不同阈值下的ap
    # p,r分别是某个类别单个阈值（取的第一个）下的值
    # ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((tp.shape[1], nc, 1000)), np.zeros((tp.shape[1], nc, 1000))

    # 取阈值列表中索引为index的阈值，输出tp,fp,p,r,f1
    # index = 2  # 0

    # 遍历每一个类别
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # 对应类别的GT数量
        n_p = i.sum()  # 对应类别的prec数量
        if n_p == 0 or n_l == 0:
            continue

        # 计算fp,tp
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # recall
        recall = tpc / (n_l + eps)  # 召回率曲线
        # np.interp：线性插值。因为recall曲线需要画1000个点，真正的点肯定少于1000，因此不存在的点，需要靠线性插值获取

        # r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)

        # precision
        precision = tpc / (tpc + fpc)

        # p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)

        # ap
        for j in range(tp.shape[1]):
            r[j][ci] = np.interp(-px, -conf[i], recall[:, j], left=0)
            p[j][ci] = np.interp(-px, -conf[i], precision[:, j], left=1)
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))

    # 注意：这里的p和r是取得第一个dis阈值的值，如果指定了dis阈值，结果是正确的。如果是按照阈值范围，那么第一个dis阈值是0，p和r的结果都是0
    f1 = 2 * p * r / (p + r + eps)
    # names = {"0": "jueyuanzi"}
    # names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    # names = dict(enumerate(names))
    names = [0]
    if plot:
        plot_pr_curve(px, py, ap, 0, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    out_tp, out_fp, out_p, out_r, out_f1 = [], [], [], [], []
    for k in range(tp.shape[1]):
        i = smooth(f1[k].mean(0), 0.1).argmax()  # 找到最大的f1索引
        p1, r1, f11 = p[k][:, i], r[k][:, i], f1[k][:, i]  # i索引对应的最大值
        tp = (r1 * nt).round()  # tp
        fp = (tp / (p1 + eps) - tp).round()  # false positives
        out_tp.append("{:.3f}".format(tp.item()))
        out_fp.append("{:.3f}".format(fp.item()))
        out_p.append("{:.3f}".format(p1.item()))
        out_r.append("{:.3f}".format(r1.item()))
        out_f1.append("{:.3f}".format(f11.item()))

    # print("tp:{}\tfp:{}\tp:{}\tr:{}\tf1:{}\tap:{}".format(tp, fp, p, r, f1, ap))
    # return tp, fp, p, r, f1, ap, unique_classes.astype(int)
    # print("tp:{}\nfp:{}\np:{}\nr:{}\nf1:{}\nap:{}".format(out_tp, out_fp, out_p, out_r, out_f1, ap))
    return out_tp, out_fp, out_p, out_r, out_f1, ap, unique_classes.astype(int)


def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = np.ones(nf // 2)  # ones padding
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return np.convolve(yp, np.ones(nf) / nf, mode='valid')  # y-smoothed


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """
    # 将0,1加入到曲线列表中
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # np.flip()：将一维数组从小到大排序
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    # interp方式计算的AP会比continuous的大
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        # 这个是求积分
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        # np.interp（d,x,y）：插值,d为待插入的横坐标，x为原先的横坐标，y为纵坐标
        # np.trapz(list,list)，求两个列表，对应点之间的四边形面积。 以定积分的形式计算AP，第一个参数是Y，第二个参数是X
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        # 这个是求面积
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def _nms(heat, kernel=3):
    hmap = F.max_pool2d(heat, kernel, stride=1, padding=kernel // 2)
    keep = (hmap == heat).float()
    return heat * keep


# img_size:(h,w)
def get_final_preds(preds, img_size, thresh=0.6, max_kp=50):
    assert len(img_size) == 2
    # preds = torch.sigmoid(preds)
    # # 关键点非极大值抑制
    # preds = _nms(preds)

    batch_size, num_joints, h, w = preds.shape
    output = []

    reshaped_preds = preds.reshape(batch_size, num_joints, -1)
    for b in range(batch_size):
        for c in range(num_joints):
            reshaped_pred = reshaped_preds[b][c]
            p = torch.where(reshaped_pred > thresh)[0]
            p = p[reshaped_pred[p].argsort(descending=True)]
            confidence = reshaped_pred[p]
            if not p.shape[0]:
                continue
            elif p.shape[0] > max_kp:
                # p = p[reshaped_pred[p].argsort(descending=True)][:max_kp]
                p = p[:max_kp]
            p_x = p % w
            p_y = torch.floor(p / w)
            t = torch.zeros(len(p), 4).to(preds)
            # 在网络输入图像上的像素x
            t[..., 0] = p_x.long() * img_size[1] / (w - 1)
            # 在网络输入图像上的像素y
            t[..., 1] = p_y.long() * img_size[0] / (h - 1)
            # 置信度
            t[..., 2] = reshaped_pred[p]
            # 关键点类别
            t[..., 3] = c

        if p.shape[0]:
            output.append(t)

    # output:[bs] 表示bs张图片的关键点信息
    # bs:[n,4] 表示这张图上有n个点，每个点有4个信息，分别是【 x y 置信度 关键点的类别】
    return output, thresh


def pp_distance(point1, point2):
    """

    :param point1: (Tensor[N, 2])
    :param point2: (Tensor[N, 2])
    :return: distance (Tensor[N, M])
    """
    # 欧几里得距离： sqrt((x1-x2)^2+(y1-y2)^2)
    #
    n = point1.shape[0]
    m = point2.shape[0]
    (x1, y1), (x2, y2) = point1[:, None].chunk(2, 2), point2.chunk(2, 1)
    return torch.sqrt(torch.pow((x1 - x2), 2) + torch.pow((y1 - y2), 2)).view(n, m)


def process_batch(precision, gts, disv):
    """
    :param precision: (Array[N, 4]), x, y conf, class
    :param gts: (Array[M, 3]), x, y, class,
    :param v:
    :return:
    """
    correct = np.zeros((precision.shape[0], disv.shape[0])).astype(bool)
    distance = pp_distance(gts[:, 0:2], precision[:, 0:2])  # [N.M] N:GT, M:prec
    # correct_class = gts[:, 2] == precision[:, 3]
    for i in range(len(disv)):
        # x = torch.where((distance <= disv[i]) & correct_class)  # distance < thresh and class match
        x = torch.where(distance <= disv[i])  # distance < thresh and class match
        if x[0].shape[0]:
            # x[0]:GT中第i个point 和 x[1]:prec中第i个point匹配上了
            matches = torch.cat((torch.stack(x, 1), distance[x[0], x[1]][:, None]),
                                dim=1).cpu().numpy()  # [n,3] gt[i] prec[i] dis
            if x[0].shape[0] > 1:
                # 按距离从小到大排序,返回匹配的索引 matches[0]:GT中点的索引  matches[1]：预测点的索引
                matches = matches[matches[:, 2].argsort()]
                # 如果一个预测点与多个GT点匹配成功，那么取第一个匹配上的预测点
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                # 如果一个GT点与多个预测点匹配成功，那么取第一个匹配上的GT点
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

            # 所以最终的结果是一对一匹配，然后将匹配上的预测点改为true，方便后续计算TP
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=disv.device)


# plot
def plot_pr_curve(px, py, ap, index=0, save_dir=Path('pr_curve.png'), names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, index]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, index].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(save_dir, dpi=250)
    plt.close()


def plot_mc_curve(px, py, save_dir=Path('mc_curve.png'), names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = smooth(py.mean(0), 0.05)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(save_dir, dpi=250)
    plt.close()


def main(opt):
    run(**vars(opt))


if __name__ == '__main__':
    run(data=r'E:\dataset\pose_estimation\insulator', weights='hrnet_best.pth',dis_thresh=2)
