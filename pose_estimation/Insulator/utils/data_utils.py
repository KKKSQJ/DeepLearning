import os
from multiprocessing.pool import ThreadPool

import cv2
import shutil
import json

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from PIL import Image
import sys
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
ROOT1 = ROOT.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

if str(ROOT1) not in sys.path:
    sys.path.append(str(ROOT1))
ROOT1 = Path(os.path.relpath(ROOT1, Path.cwd()))

# 原始数据 标注了0 和 1。需要将标签为0的区域mask掉，并且去掉标签0的点。将标签1改为标签0
def clean_data(imgs_dir, jsons_dir, out_dir):
    assert os.path.exists(imgs_dir)
    assert os.path.exists(jsons_dir)

    images_dir = os.path.join(out_dir, "images")
    annos_dir = os.path.join(out_dir, "annos")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annos_dir, exist_ok=True)

    id_list = [x.split('.')[0] for x in os.listdir(jsons_dir) if x.endswith('.json')]

    for id in tqdm(id_list):
        img = cv2.imread(os.path.join(imgs_dir, id + '.JPG'))
        anno = os.path.join(jsons_dir, id + '.json')
        with open(anno, 'r') as f:
            data = json.load(f)
            for i in range(len(data['shapes']) - 1, -1, -1):
                if data['shapes'][i]['label'] == '0':
                    points = np.array(data['shapes'][i]['points'], dtype=np.int32)
                    img = cv2.fillPoly(img, [points], color=[255, 255, 255])

                    # cv2.imshow('a', img)
                    # cv2.waitKey()
                    data['shapes'].remove(data['shapes'][i])

                elif data['shapes'][i]['label'] == '1':
                    data['shapes'][i]['label'] = '0'

            data["imagePath"] = id + ".jpg"
            cv2.imwrite(os.path.join(images_dir, id + '.jpg'), img)
            with open(os.path.join(annos_dir, id + ".json"), 'w') as f1:
                json.dump(data, f1)


# 批量重命名
def rename(imgs_dir, jsons_dir, out_dir):
    assert os.path.exists(imgs_dir)
    assert os.path.exists(jsons_dir)

    images_dir = os.path.join(out_dir, "images")
    annos_dir = os.path.join(out_dir, "annos")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annos_dir, exist_ok=True)

    id_list = [x.split('.')[0] for x in os.listdir(jsons_dir) if x.endswith('.json')]

    for i, id in enumerate(tqdm(id_list)):
        ori_img_path = os.path.join(imgs_dir, id + '.jpg')
        ori_json_path = os.path.join(jsons_dir, id + '.json')

        out_img_path = os.path.join(images_dir, "{:0>6d}.jpg".format(i))
        out_json_path = os.path.join(annos_dir, "{:0>6d}.json".format(i))

        if os.path.isfile(ori_img_path):
            shutil.copy(ori_img_path, out_img_path)
        if os.path.isfile(ori_json_path):
            with open(ori_json_path, 'r') as f:
                data = json.load(f)
                data["imagePath"] = "{:0>6d}.jpg".format(i)
                with open(out_json_path, 'w') as f1:
                    json.dump(data, f1)


# 获得数据集的均值和方差
class get_mean_and_val(object):
    def __init__(self, img_path):
        assert os.path.exists(img_path)
        self.img_path = img_path
        self.img_list = [os.path.join(self.img_path, x) for x in os.listdir(self.img_path)]
        self.cnt = 0

    def calc_channel_sum(self, img_path):
        img = np.array(Image.open(img_path).convert("RGB")) / 255.0  # RGB
        h, w, c = img.shape
        pixel_num = h * w
        channel_sum = img.sum(axis=(0, 1))
        return channel_sum, pixel_num

    def calc_channel_var(self, img_path, mean):
        img = np.array(Image.open(img_path).convert("RGB")) / 255.0
        channel_var = np.sum((img - mean) ** 2, axis=(0, 1))
        return channel_var

    def get_mean(self):
        channel_sum = np.zeros(3)
        for i, x in tqdm(enumerate(self.img_list), total=len(self.img_list)):
            c_sum, p_num = self.calc_channel_sum(x)
            channel_sum += c_sum
            self.cnt += p_num
        self.mean = channel_sum / self.cnt

        print("R_mean is %f, G_mean is %f, B_mean is %f" % (self.mean[0], self.mean[1], self.mean[2]))
        return self.mean

    def get_var(self):
        channel_sum = np.zeros(3)
        for i, x in tqdm(enumerate(self.img_list), total=len(self.img_list)):
            channel_sum += self.calc_channel_var(x, self.mean)
        self.var = np.sqrt(channel_sum / self.cnt)

        print("R_var is %f, G_var is %f, B_var is %f" % (self.var[0], self.var[1], self.var[2]))
        return self.var


# 数据增强，批量生成数据
def generate_data(imgs_dir, jsons_dir, out_dir):
    assert os.path.exists(imgs_dir)
    assert os.path.exists(jsons_dir)

    images_dir = os.path.join(out_dir, "images")
    annos_dir = os.path.join(out_dir, "annos")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annos_dir, exist_ok=True)

    # id_list = [x.split('.')[0] for x in os.listdir(jsons_dir) if x.endswith('.json')]

    from dataset import Keypoint
    from dataset.kp_transforms import AffineTransform, RandomHorizontalFlip
    from utils import draw_keypoints

    aug_nums = 25
    i = 0
    scale = (0.35, 1.35)
    ratation = (-90, 90)
    fixed_size = (762, 762)  # w h
    id = 121
    json_dict = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [],
        "imageHeight": fixed_size[1],
        "imageWidth": fixed_size[0]
    }

    dataset = Keypoint(img_path=imgs_dir, anno_path=jsons_dir)
    while i < aug_nums:
        for image, info in tqdm(dataset, total=len(dataset)):
            new_image, new_info = AffineTransform(scale=scale, rotation=ratation, fixed_size=fixed_size)(image, info)

            new_image, new_info = RandomHorizontalFlip(0.5)(new_image, new_info)
            new_image = Image.fromarray(new_image)

            # with open(r'E:\dataset\pose_estimation\insulator\annos\000000.json', 'r') as f:
            #     data = json.load(f)

            # resize_img = draw_keypoints(new_image, new_info["keypoints"], draw_text=False, r=5, font_size=20)
            new_image.save(os.path.join(images_dir, "{:>06d}.jpg".format(id)))
            # np.save(os.path.join(images_dir,"{:>06d}.jpg".format(id)),new_image)
            json_dict["imagePath"] = "{:>06d}.jpg".format(id)
            json_dict["shapes"].append({"label": "0", "points": new_info["keypoints"].tolist()})
            with open(os.path.join(annos_dir, "{:>06d}.json".format(id)), 'w') as f:
                json.dump(json_dict, f)
            json_dict["shapes"] = []

            id += 1
            # plt.imshow(resize_img)
            # plt.show()
        i += 1
        print(i)


def show_data(imgs_dir, jsons_dir):
    assert os.path.exists(imgs_dir)
    assert os.path.exists(jsons_dir)

    from dataset import Keypoint
    from dataset.kp_transforms import AffineTransform, RandomHorizontalFlip
    from utils import draw_keypoints
    scale = (0.2, 1.35)
    ratation = (-90, 90)
    fixed_size = (1024, 1024)  # w h

    dataset = Keypoint(img_path=imgs_dir, anno_path=jsons_dir)
    for image, info in tqdm(dataset, total=len(dataset)):
        # new_image, new_info = AffineTransform(scale=scale, rotation=ratation, fixed_size=fixed_size)(image, info)
        #
        # new_image, new_info = RandomHorizontalFlip(0.5)(new_image, new_info)

        resize_img = draw_keypoints(image, info["keypoints"], draw_text=False, r=5, font_size=20)
        plt.imshow(resize_img)
        plt.show()


if __name__ == '__main__':
    # img_path = r'E:\dataset\pose_estimation\Sum_7.31_120\sum_pic'
    # anno_path = r'E:\dataset\pose_estimation\Sum_7.31_120\sum_json'
    img_path = r'E:\dataset\pose_estimation\insulator\images'
    anno_path = r'E:\dataset\pose_estimation\insulator\annos'
    out_path = r'E:\dataset\pose_estimation\insulator\generate'

    # 数据清洗
    # img_path = r'E:\dataset\pose_estimation\Sum_7.31_120\sum_pic'
    # anno_path = r'E:\dataset\pose_estimation\Sum_7.31_120\sum_json'
    # clean_data(img_path, anno_path, './')

    # 数据重命名
    # rename('./images', './annos', './rename')

    # 计算数据集均值和方差
    # mean_val = get_mean_and_val(img_path)
    # means = mean_val.get_mean()
    # vars = mean_val.get_var()

    # 批量生成数据
    # img_path = r'E:\dataset\pose_estimation\insulator\images'
    # anno_path = r'E:\dataset\pose_estimation\insulator\annos'
    # out_path = r'E:\dataset\pose_estimation\insulator\generate'
    # generate_data(img_path, anno_path, out_dir=out_path)

    # 可视化标签数据
    img_path = r'E:\dataset\pose_estimation\insulator\generate\images'
    anno_path = r'E:\dataset\pose_estimation\insulator\generate\annos'
    show_data(img_path, anno_path)
