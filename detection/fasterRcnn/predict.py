import cv2
import argparse
import numpy as np
import glob
import shutil
from tqdm import tqdm
import os
import time
import json
import torch
from PIL import Image
from torchvision import transforms
from models import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from models.backbone import resnet50_fpn_backbone, MobileNetV2
from utils import draw_objs
import torchvision
import matplotlib.pyplot as plt

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']


def time_sync():
    # pytorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def create_model(name, num_classes):
    # mobileNetv2+faster_RCNN
    if name == 'mobilenetv2':
        backbone = MobileNetV2().features
        backbone.out_channels = 1280

        anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                            aspect_ratios=((0.5, 1.0, 2.0),))

        roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                        output_size=[7, 7],
                                                        sampling_ratio=2)

        model = FasterRCNN(backbone=backbone,
                           num_classes=num_classes,
                           rpn_anchor_generator=anchor_generator,
                           box_roi_pool=roi_pooler)
    elif name == "resnet50":
        # resNet50+fpn+faster_RCNN
        # 注意，这里的norm_layer要和训练脚本中保持一致
        backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
        model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


@torch.no_grad()
def run(
        model_name='resnet50',  # 网络名字
        weights='best_model.pth',  # 模型路径
        num_classes=20,  # 类别数量，不含背景
        source='./data/test',  # 测试数据路径，可以是文件夹，可以是单张图片
        use_cuda=True,  # 是否使用cuda
        view_img=False,  # 是否可视化测试图片
        save_img=True,  # 是否保存图片
        save_txt=True,  # 是否将结果保存到txt
        project='runs/result',  # 结果输出路径
        class_indices='pascal_voc_classes.json'  # json文件，存放类别和索引的关系。

):
    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")

    data_transform = transforms.Compose(
        [  # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    os.makedirs(project, exist_ok=True)

    # read class_indict
    json_path = class_indices
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    # load model
    model = create_model(name=model_name, num_classes=num_classes + 1)

    assert os.path.exists(weights), "model path: {} does not exists".format(weights)
    model.load_state_dict(torch.load(weights, map_location='cpu')["model"])
    model.eval().to(device)

    # load img
    assert os.path.exists(source), "data source: {} does not exists".format(source)
    if os.path.isdir(source):
        files = sorted(glob.glob(os.path.join(source, '*.*')))
    elif os.path.isfile(source):
        # img = Image.open(source)
        # if img.mode != 'RGB':
        #     img = img.convert('RGB')
        files = [source]
    else:
        raise Exception(f'ERROR: {source} does not exist')

    images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
    images = tqdm(images)
    image_list = []
    for img_path in images:
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # [N,C,H,W]
        # 单张图片预测
        img_tensor = data_transform(img)[None, :]

        # 多张打包成patch进行预测
        # image_list.append(data_transform(img))
        # image_tensor = torch.stack(image_list,dim=0)

        img_height, img_width = img_tensor.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t1 = time_sync()
        predictions = model(img_tensor.to(device))[0]
        t2 = time_sync()
        print("inference+NMS time: {}".format(t2 - t1))

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")
            return

        # 可视化图片
        plot_img = draw_objs(img,
                             predict_boxes,
                             predict_classes,
                             predict_scores,
                             category_index=category_index,
                             box_thresh=0.5,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)
        if view_img:
            plt.imshow(plot_img)
            plt.show()

        if save_img:
            file_name = img_path.split(os.sep)[-1].split(".")[0] + ".jpg"
            plot_img.save(os.path.join(project, file_name))

        if save_txt:
            file_name = img_path.split(os.sep)[-1].split(".")[0] + ".txt"
            f1 = open(os.path.join(project, file_name), 'w')

            box_thresh = 0.1
            idxs = np.greater(predict_scores, box_thresh)
            boxes = predict_boxes[idxs]
            classes = predict_classes[idxs]
            scores = predict_scores[idxs]
            for box, cls, score in zip(boxes, classes, scores):
                left, top, right, bottom = box
                cls = int(cls)
                score = float(score)
                f1.write("{} {} {} {} {} {}\n".format(cls, score, left, top, right, bottom))
            f1.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='resnet50')
    parser.add_argument('--weights', type=str, default='best_model.pth', help='the model path')
    parser.add_argument('--num_classes', type=int, default=20, help='num of classes')
    parser.add_argument('--source', type=str, default='./data/test', help='test data path')
    parser.add_argument('--use-cuda', type=bool, default=True)
    parser.add_argument('--view-img', type=bool, default=False)
    parser.add_argument('--save_img', type=bool, default=False)
    parser.add_argument('--save-txt', type=bool, default=True)
    parser.add_argument('--project', type=str, default='runs/result', help='output path')
    parser.add_argument('--class-indices', type=str, default='pascal_voc_classes.json',
                        help='when train,the file will generate')
    opt = parser.parse_args()
    run(**vars(opt))
