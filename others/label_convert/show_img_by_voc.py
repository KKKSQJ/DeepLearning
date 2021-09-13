import os
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from lxml import etree
from collections import defaultdict
import argparse
import sys

category_set = dict()
image_set = set()
every_class_num = defaultdict(int)

category_item_id = -1


def draw_box(img, objects, draw=True):
    for object in objects:
        category_name = object['name']
        every_class_num[category_name] += 1
        if category_name not in category_set:
            category_id = addCatItem(category_name)
        else:
            category_id = category_set[category_name]
        xmin = int(object['bndbox']['xmin'])
        ymin = int(object['bndbox']['ymin'])
        xmax = int(object['bndbox']['xmax'])
        ymax = int(object['bndbox']['ymax'])
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


def addCatItem(name):
    global category_item_id
    category_item = dict()
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    category_set[name] = category_item_id
    return category_item_id


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


def show_image(image_path, anno_path, show=False, plot_image=False):
    assert os.path.exists(image_path), "image path:{} dose not exists".format(image_path)
    assert os.path.exists(anno_path), "annotation path:{} does not exists".format(anno_path)
    anno_file_list = [os.path.join(anno_path, file) for file in os.listdir(anno_path) if file.endswith(".xml")]

    for xml_file in tqdm(anno_file_list):
        if not xml_file.endswith('.xml'):
            continue

        with open(xml_file) as fid:
            xml_str = fid.read()
        xml = etree.fromstring(xml_str)
        xml_info_dict = parse_xml_to_dict(xml)

        filename = xml_info_dict['annotation']['filename']
        image_set.add(filename)
        file_path = os.path.join(image_path, filename)
        if not os.path.exists(file_path):
            continue

        img = cv2.imread(file_path)
        if img is None:
            continue
        img = draw_box(img, xml_info_dict['annotation']['object'], show)
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
        该脚本用于voc标注格式（.xml）的标注框可视化
    参数明说：
        image_path:图片数据路径
        anno_path:xml标注文件路径
        show:是否展示标注后的图片
        plot_image:是否对每一类进行统计，并且保存图片
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-ip', '--image-path', type=str, default='./data/images', help='image path')
    parser.add_argument('-ap', '--anno-path', type=str, default='./data/labels/voc', help='annotation path')
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
        anno_path = './data/convert/voc'
        show_image(image_path, anno_path, show=True, plot_image=True)
        print(every_class_num)
        print("category nums: {}".format(len(category_set)))
        print("image nums: {}".format(len(image_set)))
        print("bbox nums: {}".format(sum(every_class_num.values())))
