from pycocotools.coco import COCO
import os
import shutil
from tqdm import tqdm
import sys
import argparse

images_nums = 0
category_nums = 0
bbox_nums = 0

# 将类别名字和id建立索引
def catid2name(coco):
    classes = dict()
    for cat in coco.dataset['categories']:
        classes[cat['id']] = cat['name']
    return classes


# 将[xmin,ymin,xmax,ymax]转换为yolo格式[x_center, y_center, w, h](做归一化)
def xyxy2xywhn(object, width, height):
    cat_id = object[0]
    xn = object[1] / width
    yn = object[2] / height
    wn = object[3] / width
    hn = object[4] / height
    out = "{} {:.5f} {:.5f} {:.5f} {:.5f}".format(cat_id, xn, yn, wn, hn)
    return out


def save_anno_to_txt(images_info, save_path):
    filename = images_info['filename']
    txt_name = filename[:-3] + "txt"
    with open(os.path.join(save_path, txt_name), "w") as f:
        for obj in images_info['objects']:
            line = xyxy2xywhn(obj, images_info['width'], images_info['height'])
            f.write("{}\n".format(line))


# 利用cocoAPI从json中加载信息
def load_coco(anno_file, xml_save_path):
    if os.path.exists(xml_save_path):
        shutil.rmtree(xml_save_path)
    os.makedirs(xml_save_path)

    coco = COCO(anno_file)
    classes = catid2name(coco)
    imgIds = coco.getImgIds()
    classesIds = coco.getCatIds()

    with open(os.path.join(xml_save_path, "classes.txt"), 'w') as f:
        for id in classesIds:
            f.write("{}\n".format(classes[id]))

    for imgId in tqdm(imgIds):
        info = {}
        img = coco.loadImgs(imgId)[0]
        filename = img['file_name']
        width = img['width']
        height = img['height']
        info['filename'] = filename
        info['width'] = width
        info['height'] = height
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        objs = []
        for ann in anns:
            object_name = classes[ann['category_id']]
            # bbox:[x,y,w,h]
            bbox = list(map(float, ann['bbox']))
            xc = bbox[0] + bbox[2] / 2.
            yc = bbox[1] + bbox[3] / 2.
            w = bbox[2]
            h = bbox[3]
            obj = [ann['category_id'], xc, yc, w, h]
            objs.append(obj)
        info['objects'] = objs
        save_anno_to_txt(info, xml_save_path)


def parseJsonFile(json_path, txt_save_path):
    assert os.path.exists(json_path), "json path:{} does not exists".format(json_path)
    if os.path.exists(txt_save_path):
        shutil.rmtree(txt_save_path)
    os.makedirs(txt_save_path)

    assert json_path.endswith('json'), "json file:{} It is not json file!".format(json_path)

    load_coco(json_path, txt_save_path)


if __name__ == '__main__':
    """
    脚本说明：
        该脚本用于将coco格式的json文件转换为yolo格式的txt文件
    参数说明：
        json_path:json文件的路径
        txt_save_path:txt保存的路径
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-jp', '--json-path', type=str, default='./data/labels/coco/train.json', help='json path')
    parser.add_argument('-s', '--save-path', type=str, default='./data/convert/yolo', help='txt save path')
    opt = parser.parse_args()

    if len(sys.argv) > 1:
        print(opt)
        parseJsonFile(opt.json_path, opt.save_path)
        # print("image nums: {}".format(images_nums))
        # print("category nums: {}".format(category_nums))
        # print("bbox nums: {}".format(bbox_nums))
    else:
        json_path = './data/labels/coco/train.json'  # r'D:\practice\compete\goodsDec\data\train\train.json'
        txt_save_path = './data/convert/yolo'
        parseJsonFile(json_path, txt_save_path)
        # print("image nums: {}".format(images_nums))
        # print("category nums: {}".format(category_nums))
        # print("bbox nums: {}".format(bbox_nums))
