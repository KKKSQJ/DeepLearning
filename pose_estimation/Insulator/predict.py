import json
import math

import matplotlib.pyplot as plt
import torch
# from torchvision import transforms
from dataset import kp_transforms as transforms
from models import HighResolution as hrnet

import cv2
import sys
import argparse
import os
from PIL import Image
import numpy as np
import glob
import time
import shutil
from tqdm import tqdm
import torch.nn.functional as F

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']


def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


# 使用max_pool进行nms，找到极大值点
def _nms(heat, kernel=3):
    hmap = F.max_pool2d(heat, kernel, stride=1, padding=kernel // 2)
    keep = (hmap == heat).float()  # 保留下极大值点
    return hmap * keep


# # 得到最终的预测结果，输出1 1 n 3 的tensor,n表示有n个点、3：（1,2）表示（x,y），3表示置信度
# def get_final_preds(pred, img_size, thresh=0.6):
#     assert len(img_size) == 2
#     # 点映射到0-1之间
#     # pred = torch.sigmoid(pred)
#     # # 使用max_pool进行nms，提出score最大点
#     # pred = _nms(pred)
#     #
#     batch_size, num_joints, h, w = pred.shape
#     reshape_pred = pred.reshape(batch_size, num_joints, -1)
#     points = torch.where(reshape_pred > thresh)[-1]
#
#     # [30 31 32 33 34 34 35 35 36 36 37 37 38 38 38 38 39 39 40 40 41 41 41 42, 42 42 42 43 43 44 44 45 46 46 47 48]
#     # [86 82 78 75 67 71 64 86 60 82 56 79 49 53 72 75 45 68 42 64 35 38 61 27, 31 54 58 23 51 44 47 40 33 37 30 26]
#     final_preds = torch.zeros((batch_size, num_joints, len(points), 3)).to(pred)
#
#     points_x = points % w
#     points_y = torch.floor(points / w)
#     final_preds[:, :, :, 2] = pred[:, :, points_y.long(), points_x.long()]
#
#     points_y = points_y * img_size[0] / (h - 1)
#     points_x = points_x * img_size[1] / (w - 1)
#     final_preds[:, :, :, 0] = points_x
#     final_preds[:, :, :, 1] = points_y
#     return final_preds, thresh

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


# 可视化关键点
def show_img(img, final_pred):
    fig = plt.figure()
    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ax = fig.add_subplot(1, 2, 1)
    for preds in final_pred:

        pos_x = round(preds[0])
        pos_y = round(preds[1])
        score = "{:.2f}".format(preds[2])
        cv2.circle(img, (int(pos_x), int(pos_y)), 1, (0, 255, 0), 1)
        # cv2.putText(img, str(score), (int(pos_x) , int(pos_y) -5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (0, 255, 0), 2)
        # plt.scatter(pos_x, pos_y, s=5, c='r')

    ax.imshow(img)
    plt.show()


# 可视化直线
def show_lines(lines, imgshow, color):
    if (lines is not None):
        for line in lines:
            rho, theta = line[0]
            ai = np.cos(theta)
            bi = np.sin(theta)
            x0 = ai * rho
            y0 = bi * rho
            x1 = int(x0 + 1000 * (-bi))
            y1 = int(y0 + 1000 * (ai))
            x2 = int(x0 - 1000 * (-bi))
            y2 = int(y0 - 1000 * (ai))
            cv2.line(imgshow, (x1, y1), (x2, y2), color, 1)


# 得到直线中的两个点，方便计算直线方程
def theta_rho_to_xxyy(theta, rho):
    ai = np.cos(theta)
    bi = np.sin(theta)
    x0 = ai * rho
    y0 = bi * rho
    x1 = int(x0 + 1000 * (-bi))
    y1 = int(y0 + 1000 * (ai))
    x2 = int(x0 - 1000 * (-bi))
    y2 = int(y0 - 1000 * (ai))
    return x1, y1, x2, y2


# 计算点到线的距离
def get_distance_points2line(point, line):
    """
    Args:
        point: [x0, y0]
        line: [x1, y1, x2, y2]
    """
    line_point1, line_point2 = np.array(line[0:2]), np.array(line[2:])
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
    return distance


# 多条直线nms得到最优直线
def nms_lines(lines, all_pts):
    counts = []
    if lines is not None:
        # 遍历直线
        for line in lines:
            # 得到直线r t
            rho, theta = line[0]
            # 得到直线上的两个点
            x1, y1, x2, y2 = theta_rho_to_xxyy(theta, rho)
            count = 0
            for pt in all_pts:
                # 计算点到直线的距离
                d = get_distance_points2line(pt, [x1, y1, x2, y2])
                if d < 9:
                    count += 1
            counts.append(count)
        # 得到点最多直线的索引
        index = np.argsort(counts)[::-1]

        output_line = []
        output_line.append(lines[index[0]])  # 找到一条最优直线
        output_line = np.array(output_line)
        np.delete(lines, index[0])  # 在所有直线中，删除这条直线
        return output_line


# 一个点pt到所有点pts的距离
def dist_to_points(pt, pts):
    dists = []
    for p in pts:
        dists.append(math.sqrt(((pt[0] - p[0]) ** 2) + ((pt[1] - p[1]) ** 2)))
    return dists


# 两个点连线的角度rho
def angle(pt1, pt2):
    a = math.atan2(-(pt2[0] - pt1[0]), (pt2[1] - pt1[1]))
    if a < 0:
        a = a + 3.14159
    return a


# 找到属于这条直线的点
def split_points(pts, line):
    lines_pts_index = []

    # 求直线上的两点
    rho = line[0][0][0]
    theta = line[0][0][1]
    x1, y1, x2, y2 = theta_rho_to_xxyy(theta, rho)

    # 根据点到直线的距离，为直线选几个初始点（种子点）
    dists = []
    for pt in pts:
        # 保存所有点到直线的距离
        dists.append(get_distance_points2line(pt, [x1, y1, x2, y2]))
    # 按照距离排序
    index = np.argsort(dists)
    # 取距离最小的前两个点  这两个点可以是直线中任意位置的点
    line_pts_index = index[0:2].tolist()

    # 根据初始点，扩充其他属于这条直线的点
    for index in line_pts_index:
        # 计算点到点的距离
        dists = dist_to_points(pts[index], pts)
        # 将点与点之间的距离进行排序
        cand_inedx = np.argsort(dists)
        # 取最近的3个点，包含它自身
        for i in range(4):
            # 如果是自身。则跳过
            if index == cand_inedx[i]:
                continue
            # 计算当前点和与他距离最近的点的连线角度
            ang0 = angle(pts[index], pts[cand_inedx[i]])
            # 计算这两个点连线的角度是否和直线的角度相似
            if abs(ang0 - theta) < 30 * 3.14 / 180.:  # 阈值30，允许偏差30度，可以调节
                if cand_inedx[i] not in line_pts_index:
                    line_pts_index.append(cand_inedx[i])
    lines_pts_index.append(line_pts_index)
    return lines_pts_index


# 利用霍夫算法，找到直线。把点分配到最优的直线
def postproc_points(img, points):
    h, w, c = img.shape
    pts = points.tolist()
    # for pt in pts:
    #     cv2.circle(img, [int(pt[0]), int(pt[1])], 3, (0, 0, 255), -1)

    # hough 变换，求直线
    image = np.zeros((h, w), dtype='uint8')
    for pt in pts:
        image[int(pt[1]), int(pt[0])] = 255
    # 3 表示一条线至少3个点
    lines = cv2.HoughLines(image, 1, np.pi / 180, 3)
    # show_lines(lines, img, (255, 255, 0))

    # nms找到一条最优直线（点最多，点到直线距离）
    # 同时，lines集合删除这条线
    line = nms_lines(lines, pts)

    #
    first_dists = []
    second_dists = []

    # 得到最优直线后，找到属于这条直线的点
    if lines is not None:
        # 得到属于这条线的点集合的索引
        list_zip = split_points(pts, line)
        colors = [(255, 0, 255), (255, 0, 0)]
        for j, line_pt in enumerate(list_zip):
            for i in line_pt:
                cv2.circle(img, (int(pts[i][0]), int(pts[i][1])), 3, colors[j % 2])
                # first_line_pts.append([pts[i][0], pts[i][1]])

    cv2.imshow("a", img[:, :, ::-1])
    cv2.waitKey()
    pass


# 画线
def draw_line(img, final_pred):
    # 这里final_pred其实就是一个 1 1 n 3
    points = final_pred.reshape(-1, 3)[:, :2]
    postproc_points(img, points)


@torch.no_grad()
def run(
        cfg='config/test.yaml',  # 配置文件
        weights='best_model.pth',  # 模型路径
        source='./data/test',  # 测试数据路径，可以是文件夹，可以是单张图片
        use_cuda=True,  # 是否使用cuda
        view_img=False,  # 是否可视化测试图片
        view_line=False,  # 将点连成线
        save_txt=True,  # 是否将结果保存到txt
        project='runs/result',  # 结果输出路径
):
    if isinstance(cfg, dict):
        config = cfg
    else:
        import yaml
        with open(cfg, encoding='utf-8') as f:
            config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    # size = config["test"]["resolution"]
    size = (448, 448)
    data_transform = transforms.Compose(
        [transforms.AffineTransform(fixed_size=size),
         transforms.KeypointToHeatMap(heatmap_hw=(size[0] // 4, size[1] // 4), keypoints_nums=1),
         transforms.ToTensor(),
         transforms.Normalize([0.616, 0.231, 0.393], [0.312, 0.288, 0.250])])
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    if save_txt:
        if os.path.exists(project):
            shutil.rmtree(project)
        os.makedirs(project)
        f = open(project + "/result.txt", 'w')

    # load model
    assert os.path.exists(weights), "model path: {} does not exists".format(weights)
    model = hrnet(base_channel=config["test"]["base_channel"], num_joint=config["test"]["num_joint"],
                  use_offset=config["test"]["use_offset"])

    model.load_state_dict(torch.load(weights, map_location="cpu")["state_dict"], strict=True)
    model.eval().to(device)

    # run once
    y = model(torch.rand(1, 3, 224, 224).to(device))

    # load img
    from dataset import Keypoint
    from torch.utils.data import DataLoader
    data_set = Keypoint(img_path=r'E:\dataset\pose_estimation\insulator\images',
                        anno_path=r'E:\dataset\pose_estimation\insulator\annos',
                        transforms=data_transform)
    dataloader = DataLoader(data_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False,
                            collate_fn=data_set.collate_fn)

    for i, (images, targets) in enumerate(dataloader):
        height = targets[0]["image_height"]
        width = targets[0]["image_width"]
        imgs = torch.stack([image.to(device) for image in images])
        t1 = time_sync()
        if config["test"]["use_offset"]:
            pred, regs = model(imgs)
        else:
            pred = model(imgs)
            regs = None

        final_pred, thresh = get_final_preds(pred, regs, (size[0], size[1]), thresh=0.65)
        t2 = time_sync()

        final_pred = final_pred[0].detach().cpu().numpy()
        img = cv2.imread(targets[0]["image_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img, new_info = transforms.AffineTransform(fixed_size=size)(img, {"box": [0, 0, width - 1, height - 1]})
        show_img(img, final_pred)
        # draw_line(img, final_pred)
        print("inference time: {:.5f}s Done.".format((t2 - t1)))

        # heatmap = targets[0]["heatmap"][None, :]
        # true_heat, thres = get_final_preds(heatmap, (size[0], size[1]), thresh=0.55)
        # show_img(img, true_heat)

    # assert os.path.exists(source), "data source: {} does not exists".format(source)
    # if os.path.isdir(source):
    #     files = sorted(glob.glob(os.path.join(source, '*.*')))
    # elif os.path.isfile(source):
    #     # img = Image.open(source)
    #     # if img.mode != 'RGB':
    #     #     img = img.convert('RGB')
    #     files = [source]
    # else:
    #     raise Exception(f'ERROR: {source} does not exist')
    #
    # images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
    # images = tqdm(images)
    # image_list = []
    # for img_path in images:
    #     img = cv2.imread(img_path)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # .transpose(0,1,2)
    #     # [N,C,H,W]
    #     # 单张图片预测
    #     height = img.shape[0]
    #     width = img.shape[1]
    #     img_tensor, target = data_transform(img, {"box": [0, 0, width - 1, height - 1]})
    #     img_tensor = img_tensor[None, :]
    #
    #     # 多张打包成patch进行预测
    #     # image_list.append(data_transform(img))
    #     # image_tensor = torch.stack(image_list,dim=0)
    #
    #     t1 = time_sync()
    #     pred = model(img_tensor.to(device))
    #     t2 = time_sync()
    #
    #     final_pred, thresh = get_final_preds(pred, (height, width), thresh=0.55)
    #     final_pred = final_pred.detach().cpu().numpy()
    #
    #     print("inference time: {:.5f}s Done.".format((t2 - t1)))
    #
    #     # 可视化图片
    #     if view_img:
    #         show_img(img, final_pred)
    #
    #     if view_line:
    #         draw_line(img, final_pred)

    #     if save_txt:
    #         file_name = img_path.split(os.sep)[-1]
    #         # f.write("{} {}\n".format(file_name, c))
    #         f.write("{} {}\n".format(file_name, class_indict[str(c)]))
    # f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/test.yaml')
    parser.add_argument('--weights', type=str, default='best.pth', help='the model path')
    parser.add_argument('--source', type=str, default='data/test', help='test data path')
    parser.add_argument('--use-cuda', type=bool, default=True)
    parser.add_argument('--view-img', type=bool, default=True)
    parser.add_argument('--view_line', type=bool, default=True)
    parser.add_argument('-s', '--save-txt', type=bool, default=True)
    parser.add_argument('--project', type=str, default='runs/result', help='output path')

    opt = parser.parse_args()
    run(**vars(opt))
