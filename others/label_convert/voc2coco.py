import xml.etree.ElementTree as ET
import os
import json
from datetime import datetime
import sys
import argparse

coco = dict()
coco['images'] = []
coco['type'] = 'instances'
coco['annotations'] = []
coco['categories'] = []

category_set = dict()
image_set = set()

category_item_id = -1
image_id = 000000
annotation_id = 0


def addCatItem(name):
    global category_item_id
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_set[name] = category_item_id
    return category_item_id


def addImgItem(file_name, size):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    image_item['license'] = None
    image_item['flickr_url'] = None
    image_item['coco_url'] = None
    image_item['date_captured'] = str(datetime.today())
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id


def addAnnoItem(object_name, image_id, category_id, bbox):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    # bbox[] is x,y,w,h
    # left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    # left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    # right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    # right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])

    annotation_item['segmentation'].append(seg)

    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)


def read_image_ids(image_sets_file):
    ids = []
    with open(image_sets_file, 'r') as f:
        for line in f.readlines():
            ids.append(line.strip())
    return ids


def parseXmlFilse(data_dir, json_save_path, split='train'):
    assert os.path.exists(data_dir), "data path:{} does not exist".format(data_dir)
    labelfile = split + ".txt"
    image_sets_file = os.path.join(data_dir, "ImageSets", "Main", labelfile)
    xml_files_list = []
    if os.path.isfile(image_sets_file):
        ids = read_image_ids(image_sets_file)
        xml_files_list = [os.path.join(data_dir, "Annotations", f"{i}.xml") for i in ids]
    elif os.path.isdir(data_dir):
        # 修改此处xml的路径即可
        # xml_dir = os.path.join(data_dir,"labels/voc")
        xml_dir = data_dir
        xml_list = os.listdir(xml_dir)
        xml_files_list = [os.path.join(xml_dir, i) for i in xml_list]

    for xml_file in xml_files_list:
        if not xml_file.endswith('.xml'):
            continue

        tree = ET.parse(xml_file)
        root = tree.getroot()

        # 初始化
        size = dict()
        size['width'] = None
        size['height'] = None

        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

        # 提取图片名字
        file_name = root.findtext('filename')
        assert file_name is not None, "filename is not in the file"

        # 提取图片 size {width,height,depth}
        size_info = root.findall('size')
        assert size_info is not None, "size is not in the file"
        for subelem in size_info[0]:
            size[subelem.tag] = int(subelem.text)

        if file_name is not None and size['width'] is not None and file_name not in image_set:
            # 添加coco['image'],返回当前图片ID
            current_image_id = addImgItem(file_name, size)
            print('add image with name: {}\tand\tsize: {}'.format(file_name, size))
        elif file_name in image_set:
            raise Exception('file_name duplicated')
        else:
            raise Exception("file name:{}\t size:{}".format(file_name, size))

        # 提取一张图片内所有目标object标注信息
        object_info = root.findall('object')
        if len(object_info) == 0:
            continue
        # 遍历每个目标的标注信息
        for object in object_info:
            # 提取目标名字
            object_name = object.findtext('name')
            if object_name not in category_set:
                # 创建类别索引
                current_category_id = addCatItem(object_name)
            else:
                current_category_id = category_set[object_name]

            # 初始化标签列表
            bndbox = dict()
            bndbox['xmin'] = None
            bndbox['xmax'] = None
            bndbox['ymin'] = None
            bndbox['ymax'] = None
            # 提取box:[xmin,ymin,xmax,ymax]
            bndbox_info = object.findall('bndbox')
            for box in bndbox_info[0]:
                bndbox[box.tag] = int(box.text)

            if bndbox['xmin'] is not None:
                if object_name is None:
                    raise Exception('xml structure broken at bndbox tag')
                if current_image_id is None:
                    raise Exception('xml structure broken at bndbox tag')
                if current_category_id is None:
                    raise Exception('xml structure broken at bndbox tag')
                bbox = []
                # x
                bbox.append(bndbox['xmin'])
                # y
                bbox.append(bndbox['ymin'])
                # w
                bbox.append(bndbox['xmax'] - bndbox['xmin'])
                # h
                bbox.append(bndbox['ymax'] - bndbox['ymin'])
                print('add annotation with object_name:{}\timage_id:{}\tcat_id:{}\tbbox:{}'.format(object_name,
                                                                                                   current_image_id,
                                                                                                   current_category_id,
                                                                                                   bbox))
                addAnnoItem(object_name, current_image_id, current_category_id, bbox)

    json_parent_dir = os.path.dirname(json_save_path)
    if not os.path.exists(json_parent_dir):
        os.makedirs(json_parent_dir)
    json.dump(coco, open(json_save_path, 'w'))
    print("class nums:{}".format(len(coco['categories'])))
    print("image nums:{}".format(len(coco['images'])))
    print("bbox nums:{}".format(len(coco['annotations'])))


if __name__ == '__main__':
    """
    脚本说明：
        本脚本用于将VOC格式的标注文件.xml转换为coco格式的标注文件.json
    参数说明：
        voc_data_dir:两种格式
            1.voc2012文件夹的路径，会自动找到voc2012/imageSets/Main/xx.txt
            2.xml标签文件存放的文件夹
        json_save_path:json文件输出的文件夹
        split:主要用于voc2012查找xx.txt,如train.txt.如果用格式2，则不会用到该参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--voc-dir', type=str, default='data/label/voc', help='voc path')
    parser.add_argument('-s', '--save-path', type=str, default='./data/convert/coco/train.json', help='json save path')
    parser.add_argument('-t', '--type', type=str, default='train', help='only use in voc2012/2007')
    opt = parser.parse_args()
    if len(sys.argv) > 1:
        print(opt)
        parseXmlFilse(opt.voc_dir, opt.save_path, opt.type)
    else:
        # voc_data_dir = r'D:\dataset\VOC2012\VOCdevkit\VOC2012'
        voc_data_dir = './data/labels/voc'
        json_save_path = './data/convert/coco/train.json'
        split = 'train'
        parseXmlFilse(data_dir=voc_data_dir, json_save_path=json_save_path, split=split)
