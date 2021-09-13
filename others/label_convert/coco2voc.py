from pycocotools.coco import COCO
import os
from lxml import etree, objectify
import shutil
from tqdm import tqdm
import sys
import argparse


# 将类别名字和id建立索引
def catid2name(coco):
    classes = dict()
    for cat in coco.dataset['categories']:
        classes[cat['id']] = cat['name']
    return classes


# 将标签信息写入xml
def save_anno_to_xml(filename, size, objs, save_path):
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder("DATA"),
        E.filename(filename),
        E.source(
            E.database("The VOC Database"),
            E.annotation("PASCAL VOC"),
            E.image("flickr")
        ),
        E.size(
            E.width(size['width']),
            E.height(size['height']),
            E.depth(size['depth'])
        ),
        E.segmented(0)
    )
    for obj in objs:
        E2 = objectify.ElementMaker(annotate=False)
        anno_tree2 = E2.object(
            E.name(obj[0]),
            E.pose("Unspecified"),
            E.truncated(0),
            E.difficult(0),
            E.bndbox(
                E.xmin(obj[1]),
                E.ymin(obj[2]),
                E.xmax(obj[3]),
                E.ymax(obj[4])
            )
        )
        anno_tree.append(anno_tree2)
    anno_path = os.path.join(save_path, filename[:-3] + "xml")
    etree.ElementTree(anno_tree).write(anno_path, pretty_print=True)


# 利用cocoAPI从json中加载信息
def load_coco(anno_file, xml_save_path):
    if os.path.exists(xml_save_path):
        shutil.rmtree(xml_save_path)
    os.makedirs(xml_save_path)

    coco = COCO(anno_file)
    classes = catid2name(coco)
    imgIds = coco.getImgIds()
    classesIds = coco.getCatIds()
    for imgId in tqdm(imgIds):
        size = {}
        img = coco.loadImgs(imgId)[0]
        filename = img['file_name']
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
        save_anno_to_xml(filename, size, objs, xml_save_path)


def parseJsonFile(data_dir, xmls_save_path):
    assert os.path.exists(data_dir), "data dir:{} does not exits".format(data_dir)

    if os.path.isdir(data_dir):
        data_types = ['train2017', 'val2017']
        for data_type in data_types:
            ann_file = 'instances_{}.json'.format(data_type)
            xmls_save_path = os.path.join(xmls_save_path, data_type)
            load_coco(ann_file, xmls_save_path)
    elif os.path.isfile(data_dir):
        anno_file = data_dir
        load_coco(anno_file, xmls_save_path)


if __name__ == '__main__':
    """
    脚本说明：
        该脚本用于将coco格式的json文件转换为voc格式的xml文件
    参数说明：
        data_dir:json文件的路径
        xml_save_path:xml输出路径
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-dir', type=str, default='./data/labels/coco/train.json', help='json path')
    parser.add_argument('-s', '--save-path', type=str, default='./data/convert/voc', help='xml save path')
    opt = parser.parse_args()
    print(opt)

    if len(sys.argv) > 1:
        parseJsonFile(opt.data_dir, opt.save_path)
    else:
        data_dir = './data/labels/coco/train.json'
        xml_save_path = './data/convert/voc'
        parseJsonFile(data_dir=data_dir, xmls_save_path=xml_save_path)
