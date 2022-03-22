import argparse
import glob
import os
import time
import json

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm

from models import fcn_resnet50

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


@torch.no_grad()
def run(
        classes=1,
        weights='best_model.pth',
        source='./data/test',
        use_cuda=True,
        view_img=False,
        save=True,
        palette_path="./utils/palette.json",
        out_path='./mask'
):
    assert os.path.exists(weights), f"weights {weights} not found."
    assert os.path.exists(source), f"image {source} not found."
    assert os.path.exists(palette_path), f"palette {palette_path} not found."
    os.makedirs(out_path, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
    print("using {} device.".format(device))

    with open(palette_path, "rb") as f:
        pallette_dict = json.load(f)
        pallette = []
        for v in pallette_dict.values():
            pallette += v

    # create model
    aux = False  # inference time not need aux_classifier
    model = fcn_resnet50(aux=aux, num_classes=classes + 1)
    # delete weights about aux_classifier
    weights_dict = torch.load(weights, map_location='cpu')['model']
    for k in list(weights_dict.keys()):
        if "aux" in k:
            del weights_dict[k]

    # load weights
    model.load_state_dict(weights_dict)
    model.to(device)
    model.eval()

    # run once
    y = model(torch.rand(1, 3, 256, 256).to(device))

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([  # transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))])

    # load img
    assert os.path.exists(source), "data source: {} does not exists".format(source)
    if os.path.isdir(source):
        files = sorted(glob.glob(os.path.join(source, '*.*')))
    elif os.path.isfile(source):
        files = [source]
    else:
        raise Exception(f'ERROR: {source} does not exist')

    images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
    images = tqdm(images)
    image_list = []
    for img_path in images:
        img_name = img_path.split(os.sep)[-1][:-4]
        original_img = Image.open(img_path)
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        mask = Image.fromarray(prediction)
        mask.putpalette(pallette)
        if view_img:
            # mask.show()
            plt.imshow(mask)
            plt.show()
        if save:
            mask.save(os.path.join(out_path, "{}.png".format(img_name)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='best_model.pth', help='the model path')
    parser.add_argument('--source', type=str, default='/data/test', help='test data path')
    parser.add_argument('--classes', type=int, default=1, help='num of classes')
    parser.add_argument('--use-cuda', type=bool, default=True)
    parser.add_argument('--view-img', type=bool, default=False)
    parser.add_argument('-s', '--save', type=bool, default=True)
    parser.add_argument('--out_path', type=str, default='runs/result', help='output path')
    parser.add_argument('--palette_path', type=str, default='palette.json')
    opt = parser.parse_args()
    run(**vars(opt))
