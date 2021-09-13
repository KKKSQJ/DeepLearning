import argparse
import os
import sys
from collections import defaultdict
from xml import etree
from pycocotools.coco import COCO

import cv2
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm import tqdm

category_set = dict()
image_set = set()
every_class_num = defaultdict(int)

category_item_id = -1


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
        xmin = int(object[1])
        ymin = int(object[2])
        xmax = int(object[3])
        ymax = int(object[4])
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


# 将类别名字和id建立索引
def catid2name(coco):
    classes = dict()
    for cat in coco.dataset['categories']:
        classes[cat['id']] = cat['name']
    return classes


def show_image(image_path, anno_path, show=False, plot_image=False):
    assert os.path.exists(image_path), "image path:{} dose not exists".format(image_path)
    assert os.path.exists(anno_path), "annotation path:{} does not exists".format(anno_path)
    if not anno_path.endswith(".json"):
        raise RuntimeError("ERROR {} dose not a json file".format(anno_path))

    coco = COCO(anno_path)
    classes = catid2name(coco)
    imgIds = coco.getImgIds()
    classesIds = coco.getCatIds()
    for imgId in tqdm(imgIds):
        size = {}
        img = coco.loadImgs(imgId)[0]
        filename = img['file_name']
        image_set.add(filename)
        width = img['width']
        height = img['height']
        size['width'] = width
        size['height'] = height
        size['depth'] = 3
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        objs = []
        for ann in anns:
            object_name = classes[ann['category_id']]
            # bbox:[x,y,w,h]
            bbox = list(map(int, ann['bbox']))
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[0] + bbox[2]
            ymax = bbox[1] + bbox[3]
            obj = [object_name, xmin, ymin, xmax, ymax]
            objs.append(obj)

        file_path = os.path.join(image_path, filename)
        img = cv2.imread(file_path)
        if img is None:
            continue
        img = draw_box(img, objs, show)
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
        该脚本用于coco标注格式（.json）的标注框可视化
    参数明说：
        image_path:图片数据路径
        anno_path:json标注文件路径
        show:是否展示标注后的图片
        plot_image:是否对每一类进行统计，并且保存图片
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--image-path', type=str, default='./data/images', help='image path')
    parser.add_argument('-ap', '--anno-path', type=str, default='./data/labels/coco/train.json', help='annotation path')
    parser.add_argument('-s', '--show', action='store_true', help='weather show img')
    parser.add_argument('-p', '--plot-image', action='store_true')
    opt = parser.parse_args()

    if len(sys.argv) > 1:
        print(opt)
        show_image(opt.image_path, opt.anno_path, opt.show, opt.plot_image)
        print(every_class_num)
        print("category nums: {}".format(len(category_set)))
        print("image nums: {}".format(len(image_set)))
        print("bbox nums: {}".format(sum(every_class_num.values())))
    else:
        image_path = './data/images'
        anno_path = './data/labels/coco/train.json'
        show_image(image_path, anno_path, show=True, plot_image=True)
        print(every_class_num)
        print("category nums: {}".format(len(category_set)))
        print("image nums: {}".format(len(image_set)))
        print("bbox nums: {}".format(sum(every_class_num.values())))
