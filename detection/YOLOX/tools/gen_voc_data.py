import os
import argparse
import sys
import shutil
from tqdm import tqdm
import cv2


def run(opt):
    in_path = opt.in_path
    out_path = opt.out_path

    image_path = os.path.join(in_path, "images")
    anno_path = os.path.join(in_path, "annotations")
    assert os.path.exists(image_path), f"ERROR :{image_path} does not exists"
    assert os.path.exists(anno_path), f"ERROR :{anno_path} does not exists"

    train_img_path = os.path.join(out_path, "train2007", "JPEGImages")
    train_anno_path = os.path.join(out_path, "train2007", "Annotations")
    test_img_path = os.path.join(out_path, "test2007", "JPEGImages")
    test_anno_path = os.path.join(out_path, "test2007", "Annotations")
    for path in [train_img_path,train_anno_path,test_img_path,test_anno_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    with open(os.path.join(in_path, "train.txt"),'r') as f:
        for i in tqdm(f.readlines()):
            info = i.strip().split(" ")
            img_name, xml_name = (x.split("/")[-1] for x in info)
            shutil.copy(os.path.join(image_path, img_name), os.path.join(train_img_path, img_name))
            shutil.copy(os.path.join(anno_path, xml_name), os.path.join(train_anno_path, xml_name))
            if not img_name.split(".")[-1].lower() in ["jpg", "jpeg"]:
                pass

    with open(os.path.join(in_path, "valid.txt"),'r') as f:
        for i in tqdm(f.readlines()):
            info = i.strip().split(" ")
            img_name, xml_name = (x.split("/")[-1] for x in info)
            shutil.copy(os.path.join(image_path, img_name), os.path.join(test_img_path, img_name))
            shutil.copy(os.path.join(anno_path, xml_name), os.path.join(test_anno_path, xml_name))
            if not img_name.split(".")[-1].lower() in ["jpg", "jpeg"]:
                pass

    classes = set()
    with open(os.path.join(in_path, "label_list.txt"), 'r') as f:
        for i in tqdm(f.readlines()):
            classes.add(i.strip())
    print(classes)






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-path',default=r'D:\dataset\roadsign_voc')
    parser.add_argument('--out-path',default=r'D:\dataset\roadsign_voc\voc')
    opt = parser.parse_args()
    print(opt)
    run(opt)

if __name__ == '__main__':
    main()

