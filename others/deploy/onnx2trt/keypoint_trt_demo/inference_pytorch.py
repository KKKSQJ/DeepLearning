import time
import torch
from hrnet import HighResolution as hrnet

import numpy as np
import os
import glob
from tqdm import tqdm
import cv2

IMG_FORMATS = ["jpg", "png", "jpeg"]


def data_preprocessing(file_path, size=(224, 224), mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]):
    # H W C BGR
    img = cv2.imread(file_path)
    # RESIZE
    img = cv2.resize(img, dsize=tuple(size))
    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalization
    mean = np.array(mean).astype(np.float32)
    std = np.array(std).astype(np.float32)
    img = (img / 255. - mean) / std
    # B H W C
    img = np.expand_dims(img, axis=0)
    # B H W C -> B C H W
    img = np.transpose(img, (0, 3, 1, 2)).astype(np.float32)
    return img


# 得到最终的预测结果，输出1 1 n 3 的tensor,n表示有n个点、3：（1,2）表示（x,y），3表示置信度
def get_keypoints_preds(preds, img_size, thresh=0.6, max_kp=50):
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


def draw_pic(image, points):
    for i, p in enumerate(points):
        pos_x = int(p.cpu().numpy()[0])
        pos_y = int(p.cpu().numpy()[1])
        conf = float(p.cpu().numpy()[2])
        cv2.circle(image, (pos_x, pos_y), 1, (0, 255, 0), 2)


#  b c h w
def decode_image(image, mean, std):
    img = image.squeeze().transpose(1, 2, 0)
    # *std+mean * 255
    img = (img * std + mean) * 255.
    # bgr->rgb
    img = img[:, :, ::-1]
    img = img.astype(np.uint8)
    return img


@torch.no_grad()
def inference(
        weight,
        data_source,
        size=(448, 448),
        mean=[0.406, 0.456, 0.485],
        std=[0.225, 0.224, 0.229],
        show_img=False,
        cal_fps=False
):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model
    model = hrnet(base_channel=32, num_joint=1)
    ckpt = torch.load(weight, map_location='cpu')["state_dict"]
    model.load_state_dict(ckpt, strict=True)
    model = model.to(device)
    model.eval()

    # Load img
    assert os.path.exists(data_source), "data source: {} does not exists".format(data_source)
    if os.path.isdir(data_source):
        files = sorted(glob.glob(os.path.join(data_source, '*.*')))
    elif os.path.isfile(data_source):
        files = [data_source]
    else:
        raise Exception(f'ERROR: {data_source} does not exist')

    images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
    # images = tqdm(images)

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    fps = 0
    pure_inf_time = 0
    if cal_fps:
        if len(images) < num_warmup:
            images *= 100

    for index, image_file in (enumerate(images)):
        # torch.cuda.synchronize()
        image = data_preprocessing(image_file, size=size, mean=mean, std=std)

        t1 = time.perf_counter()
        output = model(torch.from_numpy(image).to(device))
        out, _ = get_keypoints_preds(output, img_size=size, thresh=0.6, max_kp=50)
        t2 = time.perf_counter()
        elapsed = t2 - t1

        # fps 计算，如果不带后处理，fps为22.8。如果带了后处理，fps为9.2

        draw_img = decode_image(image, mean, std)
        if show_img and len(out[0]) > 0:
            draw_pic(draw_img, out[0])
            cv2.imshow('img', draw_img)
            cv2.waitKey()

        # fps
        if index >= num_warmup:
            pure_inf_time += elapsed

            if (index + 1) % 100 == 0:
                fps = (index + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{index + 1:<3}/ {len(images)}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (index + 1) == len(images) and (index + 1) > num_warmup:
            fps = (index + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                flush=True)

        print("time:{}".format(t2 - t1))
        # print(output)


def parser_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='model.pth')
    parser.add_argument('--data_source', type=str, default='000000.jpg')  # 测试数据源，可以是单张图片，可以是文件夹
    parser.add_argument('--show_img', action='store_true', default=False)
    parser.add_argument('--cal_fps', action='store_true', default=False)
    return parser.parse_args()


if __name__ == '__main__':
    # inference(
    #     weight='resnet50-0676ba61.pth',
    #     data_source='goldfish_class_1.jpg',
    #     size=(224, 224),
    #     mean=[0.406, 0.456, 0.485],
    #     std=[0.225, 0.224, 0.229],
    #     cal_fps=True
    # )

    args = parser_args()
    inference(
        weight=args.weight,
        data_source=args.data_source,
        size=(448, 448),
        mean=[0.406, 0.456, 0.485],
        std=[0.225, 0.224, 0.229],
        show_img=False,
        cal_fps=args.cal_fps
    )
