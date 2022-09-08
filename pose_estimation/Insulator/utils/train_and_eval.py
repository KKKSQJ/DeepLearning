import time
import torch
import numpy as np
import torch.nn.functional as F
from utils.torch_utils import AverageMeter, ProgressMeter, warmup_lr_scheduler, _tranpose_and_gather_feature, \
    _gather_feature
from utils.loss import _reg_loss
import logging
import matplotlib.pyplot as plt
from pathlib import Path


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


# img_size:(h,w)
def get_final_preds(preds, regs, img_size, thresh=0.6, max_kp=50):
    assert len(img_size) == 2
    # preds = torch.sigmoid(preds)
    # # 关键点非极大值抑制
    # preds = _nms(preds)

    batch_size, num_joints, h, w = preds.shape
    output = []

    reshaped_preds = preds.reshape(batch_size, num_joints, -1)
    reshaped_regs = regs.reshape(batch_size, 2, -1)
    for b in range(batch_size):
        for c in range(num_joints):
            reshaped_pred = reshaped_preds[b][c]
            reshaped_reg = reshaped_regs[b]
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
            if regs is not None:
                offset_x = reshaped_reg[0][p_x.long()]
                offset_y = reshaped_reg[1][p_y.long()]
            else:
                offset_x = 0.
                offset_y = 0.

            t = torch.zeros(len(p), 4).to(preds)
            # 在网络输入图像上的像素x
            t[..., 0] = (p_x + offset_x).float() * img_size[1] / (w - 1)
            # 在网络输入图像上的像素y
            t[..., 1] = (p_y + offset_y).float() * img_size[0] / (h - 1)
            # 置信度
            t[..., 2] = reshaped_pred[p]
            # 关键点类别
            t[..., 3] = c

        if p.shape[0]:
            output.append(t)

    # output:[bs] 表示bs张图片的关键点信息
    # bs:[n,4] 表示这张图上有n个点，每个点有4个信息，分别是【 x y 置信度 关键点的类别】
    return output, thresh


# 和get_final_preds作用一样
def pred_decode(hmap, regs, k):
    """
    :param hmap: 预测的特征图 B C H W
    :param regs: 预测的偏移量 B 2 H W
    :param k:    取前top-k个点
    :return:
    """

    batch, c, height, width = hmap.shape
    # hmap = torch.sigmoid(hmap)
    # # 关键点非极大值抑制
    # hmap = _nms(hmap)

    scores, inds, clses, ys, xs = _top_k(hmap, k=k)

    # from [bs c h w] to [bs, h, w, c]
    regs = _tranpose_and_gather_feature(regs, inds)

    regs = regs.view(batch, k, 2)

    xs = xs.view(batch, k, 1) + regs[:, :, 0:1]
    ys = ys.view(batch, k, 1) + regs[:, :, 1:2]

    clses = clses.view(batch, k, 1).float()
    scores = scores.view(batch, k, 1)

    return


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


# 使用max_pool进行nms，找到极大值点
def _nms(heat, kernel=3):
    hmap = F.max_pool2d(heat, kernel, stride=1, padding=kernel // 2)
    keep = (hmap == heat).float()  # 保留下极大值点
    return hmap * keep


# 找个top-k个关键点
def _top_k(score, k=100):
    batch, c, height, width = score.shape

    topk_score, topk_inds = torch.topk(score.view(batch, c, -1), k)
    '''
    >>> torch.topk(torch.from_numpy(x), 3)
    torch.return_types.topk(
    values=tensor([[0.4863, 0.2339, 0.1153],
            [0.8370, 0.5961, 0.2733],
            [0.7303, 0.4656, 0.4475]], dtype=torch.float64),
    indices=tensor([[3, 2, 0],
            [1, 0, 2],
            [2, 1, 3]]))
    >>> x
    array([[0.11530714, 0.014376  , 0.23392263, 0.48629663],
        [0.59611302, 0.83697236, 0.27330404, 0.17728915],
        [0.36443852, 0.46562404, 0.73033529, 0.44751189]])
    '''

    topk_inds = topk_inds % (height * width)
    # 找到横坐标
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # to shape:[batch, class*h*w]
    topk_score, topk_ind = torch.topk(topk_score.view(batch, -1), k)

    # 所有类别中找到最大值
    topk_clses = (topk_ind / k).int()

    topk_inds = _gather_feature(topk_inds.view(
        batch, -1, 1), topk_ind).view(batch, k)

    topk_ys = _gather_feature(topk_ys.view(
        batch, -1, 1), topk_ind).view(batch, k)

    topk_xs = _gather_feature(topk_xs.view(
        batch, -1, 1), topk_ind).view(batch, k)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


# train one in epoch
def train_one_epoch(model, data_loader, epoch, optimizer, device, RANK, loss_function, tb_writer, warmup=False,
                    config=None):
    model.train()

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.6f')
    lr = AverageMeter("Lr", ':.6f')

    if RANK != -1:
        # train_sampler.set_epoch(epoch)
        data_loader.sampler.set_epoch(epoch)

    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses, lr],
        prefix="Epoch: [{}]".format(epoch))

    lr_scheduler = None
    if epoch == 0 and warmup is True:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    optimizer.zero_grad()

    end = time.time()
    for i, (images, targets) in enumerate(data_loader):
        data_time.update(time.time() - end)
        ni = i + len(data_loader) * epoch
        imgs = torch.stack([image.to(device) for image in images])

        if config["train"]["use_offset"]:
            gt_regs = torch.stack([(t["regs"]).to(device) for t in targets])
            ind_masks = torch.stack([(t["inds_masks"]).to(device) for t in targets])
            inds = torch.stack([(t["inds"]).to(device) for t in targets])
            pred, regs = model(imgs)

            regs = [_tranpose_and_gather_feature(regs, inds)]

            loss = 10 * loss_function(pred, targets) + 10 * _reg_loss(regs, gt_regs, ind_masks)

        else:
            pred = model(imgs)
            loss = loss_function(pred, targets) * 10
        loss.backward()

        losses.update(loss.item(), imgs.size(0))

        optimizer.step()
        optimizer.zero_grad()

        if lr_scheduler is not None:
            lr_scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        lr.update(optimizer.param_groups[0]['lr'])
        if RANK in {-1, 0}:
            if i % config["train"]["print_freq"] == 0:
                progress.display(i)
            tb_writer.add_scalar("train_lr", optimizer.param_groups[0]["lr"], ni)
    return losses.avg, optimizer.param_groups[0]['lr']


@torch.no_grad()
def key_point_eval(model, data_loader, epoch, optimizer, device, loss_function, tb_writer, config):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':6.4f')
    AP = AverageMeter('Ap', ':6.4f')
    R = AverageMeter('recall', ':6.4f')
    P = AverageMeter('precision', ':6.4f')
    F1 = AverageMeter('f1', ':6.4f')

    progress = ProgressMeter(
        len(data_loader),
        [batch_time, losses, P, R, F1, AP],
        prefix='Test: ')

    model.eval()
    conf_thresh = config["train"]["conf_thresh"]
    dis_thresh = config["train"]["dis_thresh"]

    conf_thresh = torch.linspace(0.5, 0.99, 50, device=device) if conf_thresh is None else torch.ones(1,
                                                                                                      device=device) * conf_thresh
    nconf = conf_thresh.numel()
    disv = torch.linspace(0, 10, 10, device=device) if dis_thresh is None else torch.ones(1, device=device) * dis_thresh
    ndis = disv.numel()

    total_images = 0
    total_gts = 0
    total_precs = 0
    stats = []

    with torch.no_grad():
        end = time.time()

        for i, (imgs, targets) in enumerate(data_loader):
            total_images += imgs.shape[0]
            imgs = torch.stack([image.to(device) for image in imgs])

            if config["train"]["use_offset"]:
                gt_regs = torch.stack([(t["regs"]).to(device) for t in targets])
                ind_masks = torch.stack([(t["inds_masks"]).to(device) for t in targets])
                inds = torch.stack([(t["inds"]).to(device) for t in targets])
                preds, pre_regs = model(imgs)
                regs = pre_regs.clone()
                regs = [_tranpose_and_gather_feature(regs, inds)]
                loss = 10 * loss_function(preds, targets) + 10 * _reg_loss(regs, gt_regs, ind_masks)
            else:
                preds = model(imgs)
                loss = loss_function(preds, targets) * 10

            # preds = model(imgs)
            # loss = loss_function(preds, targets)
            losses.update(loss.item(), imgs.size(0))

            # get final points
            imgsz = config["train"]["resolution"]
            out, _ = get_final_preds(preds,
                                     regs=pre_regs if config["train"]["use_offset"] else None,
                                     img_size=imgsz,
                                     thresh=0.6,
                                     max_kp=50)

            # out1 = pred_decode(preds,regs,k=50)

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
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, plot=False)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if len(stats) and stats[0].any():
            P.update(float(p[0]))
            R.update(float(r[0]))
            F1.update(float(f1[0]))
            AP.update(float(ap[0][0]))

            # if i % config["train"]["print_freq"] == 0:
            #     progress.display(i)
        progress.display(i)

    tags = ["test loss", "p", "r", "f1" "ap", "learning_rate"]
    tb_writer.add_scalar(tags[0], losses.avg, epoch)
    tb_writer.add_scalar(tags[1], P.avg, epoch)
    tb_writer.add_scalar(tags[2], R.avg, epoch)
    tb_writer.add_scalar(tags[3], F1.avg, epoch)
    tb_writer.add_scalar(tags[4], AP.avg, epoch)
    # tb_writer.add_scalar(tags[5], optimizer.param_groups[0]["lr"], epoch)

    # logging.info("Test Epoch\t\tloss\t\tap\t\tLr")
    # logging.info("{}\t\t{:.3}\t\t{:.3}\t\t{:.3}".format(epoch, losses.avg, losses.avg, optimizer.param_groups[0]["lr"]))
    pf = '%5s' + '%11i' * 1 + '%11.3g' * 5  # print format
    logging.info(pf % ('Eval', total_images, P.avg, R.avg, F1.avg, AP.avg, conf_thresh.detach().item()))
    return losses.avg, AP.avg
