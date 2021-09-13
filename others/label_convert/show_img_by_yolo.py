import argparse
import os
import sys
from collections import defaultdict

import cv2
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm

category_set = dict()
image_set = set()
every_class_num = defaultdict(int)

category_item_id = -1


def xywhn2xyxy(box, size):
    box = list(map(float, box))
    size = list(map(float, size))
    xmin = (box[0] - box[2] / 2.) * size[0]
    ymin = (box[1] - box[3] / 2.) * size[1]
    xmax = (box[0] + box[2] / 2.) * size[0]
    ymax = (box[1] + box[3] / 2.) * size[1]
    return (xmin, ymin, xmax, ymax)


def addCatItem(name):
    global category_item_id
    category_item = dict()
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    category_set[name] = category_item_id
    return category_item_id


def draw_box(img, objects, draw=True):
    for object in objects:
        category_name = object[0]
        every_class_num[category_name] += 1
        if category_name not in category_set:
            category_id = addCatItem(category_name)
        else:
            category_id = category_set[category_name]
        xmin = int(object[1][0])
        ymin = int(object[1][1])
        xmax = int(object[1][2])
        ymax = int(object[1][3])
        if draw:
            def hex2rgb(h):  # rgb order (PIL)
                return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

            hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                   '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')

            palette = [hex2rgb('#' + c) for c in hex]
            n = len(palette)
            c = palette[int(category_id) % n]
            bgr = False
            color = (c[2], c[1], c[0]) if bgr else c

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color)
            cv2.putText(img, category_name, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, color)
    return img


def show_image(image_path, anno_path, show=False, plot_image=False):
    assert os.path.exists(image_path), "image path:{} dose not exists".format(image_path)
    assert os.path.exists(anno_path), "annotation path:{} does not exists".format(anno_path)
    anno_file_list = [os.path.join(anno_path, file) for file in os.listdir(anno_path) if file.endswith(".txt")]

    with open(anno_path + "/classes.txt", 'r') as f:
        classes = f.readlines()

    category_id = dict((k, v.strip()) for k, v in enumerate(classes))

    for txt_file in tqdm(anno_file_list):
        if not txt_file.endswith('.txt') or 'classes' in txt_file:
            continue
        filename = txt_file.split(os.sep)[-1][:-3] + "jpg"
        image_set.add(filename)
        file_path = os.path.join(image_path, filename)
        if not os.path.exists(file_path):
            continue

        img = cv2.imread(file_path)
        if img is None:
            continue
        width = img.shape[1]
        height = img.shape[0]

        objects = []
        with open(txt_file, 'r') as fid:
            for line in fid.readlines():
                line = line.strip().split()
                category_name = category_id[int(line[0])]
                bbox = xywhn2xyxy((line[1], line[2], line[3], line[4]), (width, height))
                obj = [category_name, bbox]
                objects.append(obj)

        img = draw_box(img, objects, show)
        if show:
            cv2.imshow(filename, img)
            cv2.waitKey()
            cv2.destroyAllWindows()
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(every_class_num)), every_class_num.values(), align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(every_class_num)), every_class_num.keys(), rotation=90)
        # 在柱状图上添加数值标签
        for index, (i, v) in enumerate(every_class_num.items()):
            plt.text(x=index, y=v, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('class distribution')

        plt.savefig("class_distribution.png")
        plt.show()


if __name__ == '__main__':
    """
    脚本说明：
        该脚本用于yolo标注格式（.txt）的标注框可视化
    参数明说：
        image_path:图片数据路径
        anno_path:txt标注文件路径
        show:是否展示标注后的图片
        plot_image:是否对每一类进行统计，并且保存图片
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--image-path', type=str, default='./data/images', help='image path')
    parser.add_argument('-ap', '--anno-path', type=str, default='./data/labels/yolo', help='annotation path')
    parser.add_argument('-s', '--show', action='store_true', help='weather show img')
    parser.add_argument('-p', '--plot-image', action='store_true')
    opt = parser.parse_args()

    if len(sys.argv) > 1:
        print(opt)
        show_image(opt.image_path, opt.anno_path, opt.show, opt.plot_image)
    else:
        image_path = './data/images'
        anno_path = './data/labels/yolo'
        show_image(image_path, anno_path, show=True, plot_image=True)
        print(every_class_num)
        print("category nums: {}".format(len(category_set)))
        print("image nums: {}".format(len(image_set)))
        print("bbox nums: {}".format(sum(every_class_num.values())))
