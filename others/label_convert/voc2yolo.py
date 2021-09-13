import os
import json
import argparse
import sys
import shutil
from lxml import etree
from tqdm import tqdm

category_set = set()
image_set = set()
bbox_nums = 0


def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args:
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """
    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def write_classIndices(category_set):
    class_indices = dict((k, v) for v, k in enumerate(category_set))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)


def xyxy2xywhn(bbox, size):
    bbox = list(map(float, bbox))
    size = list(map(float, size))
    xc = (bbox[0] + (bbox[2] - bbox[0]) / 2.) / size[0]
    yc = (bbox[1] + (bbox[3] - bbox[1]) / 2.) / size[1]
    wn = (bbox[2] - bbox[0]) / size[0]
    hn = (bbox[3] - bbox[1]) / size[1]
    return (xc, yc, wn, hn)


def parser_info(info: dict, only_cat=True, class_indices=None):
    filename = info['annotation']['filename']
    image_set.add(filename)
    objects = []
    width = int(info['annotation']['size']['width'])
    height = int(info['annotation']['size']['height'])
    for obj in info['annotation']['object']:
        obj_name = obj['name']
        category_set.add(obj_name)
        if only_cat:
            continue
        xmin = int(obj['bndbox']['xmin'])
        ymin = int(obj['bndbox']['ymin'])
        xmax = int(obj['bndbox']['xmax'])
        ymax = int(obj['bndbox']['ymax'])
        bbox = xyxy2xywhn((xmin, ymin, xmax, ymax), (width, height))
        if class_indices is not None:
            obj_category = class_indices[obj_name]
            object = [obj_category, bbox]
            objects.append(object)

    return filename, objects


def parseXmlFilse(voc_dir, save_dir):
    assert os.path.exists(voc_dir), "ERROR {} does not exists".format(voc_dir)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    xml_files = [os.path.join(voc_dir, i) for i in os.listdir(voc_dir) if os.path.splitext(i)[-1] == '.xml']
    for xml_file in xml_files:
        with open(xml_file) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        info_dict = parse_xml_to_dict(xml)
        parser_info(info_dict, only_cat=True)

    with open(save_dir + "/classes.txt", 'w') as classes_file:
        for cat in sorted(category_set):
            classes_file.write("{}\n".format(cat))

    class_indices = dict((v, k) for k, v in enumerate(sorted(category_set)))

    xml_files = tqdm(xml_files)
    for xml_file in xml_files:
        with open(xml_file) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        info_dict = parse_xml_to_dict(xml)
        filename, objects = parser_info(info_dict, only_cat=False, class_indices=class_indices)
        if len(objects) != 0:
            global bbox_nums
            bbox_nums += len(objects)
            with open(save_dir + "/" + filename.split(".")[0] + ".txt", 'w') as f:
                for obj in objects:
                    f.write(
                        "{} {:.5f} {:.5f} {:.5f} {:.5f}\n".format(obj[0], obj[1][0], obj[1][1], obj[1][2], obj[1][3]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc-dir', type=str, default='./data/labels/voc')
    parser.add_argument('--save-dir', type=str, default='./data/convert/yolo')
    opt = parser.parse_args()
    if len(sys.argv) > 1:
        print(opt)
        parseXmlFilse(**vars(opt))
        print("image nums: {}".format(len(image_set)))
        print("category nums: {}".format(len(category_set)))
        print("bbox nums: {}".format(bbox_nums))
    else:
        voc_dir = './data/labels/voc'
        save_dir = './data/convert/yolo'
        parseXmlFilse(voc_dir, save_dir)
        print("image nums: {}".format(len(image_set)))
        print("category nums: {}".format(len(category_set)))
        print("bbox nums: {}".format(bbox_nums))
