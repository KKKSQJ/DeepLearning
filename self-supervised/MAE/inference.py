"""
    def _load_mae_pretrain(self):
        state_dict = torch.load("weights/vit-mae_losses_0.20791142220139502.pth", map_location="cpu")['state_dict']
        ckpt_state_dict = {}
        for key, value in state_dict.items():
            if 'Encoder.' in key:
                if key[8:] in self.model.state_dict().keys():
                    ckpt_state_dict[key[8:]] = value

        for key, value in self.model.state_dict().items():
            if key not in ckpt_state_dict.keys():
                print('There only the FC have no load pretrain!!!', key)

        state = self.model.state_dict()
        state.update(ckpt_state_dict)
        self.model.load_state_dict(state)
        print("model load the mae pretrain!!!")
"""
import glob
import os
from pathlib import Path
from tqdm import tqdm

import torch
import argparse
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from models.MAE import MAEVisonTransformer as MAE
from torch.cuda.amp import autocast as autocast
import numpy as np
from torchvision.transforms import transforms


def parser_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./assets', help='img path or img dir')
    parser.add_argument('--weights', type=str, default='./checkpoints/best-vit-mae.pth')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--use-gpu', type=int, default=1, help='0:cpu, 1:gpu')
    parser.add_argument('--result-dir', type=str, default='./result', help='path of result dir')

    args = parser.parse_args()
    return args


def run(args):
    assert os.path.exists(args.weights)
    os.makedirs(args.result_dir, exist_ok=True)
    p = str(Path(args.source).absolute())  # os-agnostic absolute path
    # 如果采用正则化表达式提取图片，直接使用glob获取文件路径
    if '*' in p:
        files = sorted(glob.glob(p, recursive=True))  # glob
    # 如果path是一个文件夹，使用glob获取全部文件路径
    elif os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    # 是文件则直接获取
    elif os.path.isfile(p):
        files = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')

    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')

    model = MAE(
        image_size=args.img_size,
        patch_size=16,
        encoer_dim=768,
        mlp_dim=1024,
        encoder_depth=12,
        num_encoder_head=12,
        dim_per_head=64,
        decoder_dim=512,
        decoder_depth=8,
        num_decoder_head=16,
        mask_ratio=0.75
    )

    ckpt = torch.load(args.weights, map_location='cpu')['state_dict']
    model.load_state_dict(ckpt, strict=True)
    model = model.to(device)
    model.eval()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)]
    )

    for file in tqdm(files):
        name = file.split(os.sep)[-1].split('.')[0]
        img_raw = Image.open(file)
        if img_raw.mode != "RGB":
            img_raw = img_raw.convert("RGB")
        h, w = img_raw.height, img_raw.width
        ratio = h / w
        print(f"image hxw: {h} x {w} mode: {img_raw.mode}")

        img_size = args.img_size if isinstance(args.img_size, tuple) else (args.img_size, args.img_size)
        img = img_raw.resize(img_size)
        rh, rw = img.height, img.width
        print(f'resized image hxw: {rh} x {rw} mode: {img.mode}')

        img.save(os.path.join(args.result_dir, f'src_{name}_{img_size}.jpg'))

        #img_ts = transform(img).unsqueeze(0).to(device)
        img_ts = ToTensor()(img).unsqueeze(0).to(device)
        print(f"input tensor shape: {img_ts.shape} dtype: {img_ts.dtype} device: {img_ts.device}")

        with torch.no_grad():
            # with autocast:
            recons_img_ts, masked_img_ts = model.predict(img_ts)
            recons_img_ts, masked_img_ts = recons_img_ts.cpu().squeeze(0), masked_img_ts.cpu().squeeze(0)
            # recons_img_ts, masked_img_ts = recons_img_ts.cpu().numpy().squeeze(0)*std+mean, masked_img_ts.cpu().numpy().squeeze(0)*std+mean
            # 将结果保存下来以便和原图比较
            recons_img = ToPILImage()(recons_img_ts)
            recons_img.save(os.path.join(args.result_dir, f'recons_{name}.jpg'))

            masked_img = ToPILImage()(masked_img_ts)
            masked_img.save(os.path.join(args.result_dir, f'masked_{name}.jpg'))


if __name__ == '__main__':
    args = parser_arg()
    run(args)
