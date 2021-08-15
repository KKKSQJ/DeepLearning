from model import resnet18, resnet34, vgg16, alexnet
import torch
import numpy as np
from torchvision import transforms
import os
from PIL import Image
import matplotlib.pyplot as plt

def init_transform(name:str='resnet'):
    if name == 'resnet':
        data_transform = transforms.Compose(
            [
                transforms.Resize(256),  # 单个参数，最小边resize到256,另一边保持比例缩放
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
    elif name == 'alexnet':
        data_transform = transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
    elif name == 'vgg':
        data_transform = transforms.Compose(
            [
                transforms.Resize(256),  # 单个参数，最小边resize到256,另一边保持比例缩放
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
    else:
        raise ValueError("transformer name must be 'resnet' or 'alexnet' or 'vgg'.now transformer type is {}".format(name))

    return data_transform

def init_model(name:str='resnet',weights:str='./resnet34.pth'):
    assert os.path.exists(weights),"weights path: {} is not exists".format(weights)
    if name == 'resnet':
        model = resnet34(num_classes=1000)
        model.load_state_dict(torch.load(weights))
    elif name == 'alexnet':
        model = alexnet(num_classes=1000)
        model.load_state_dict(torch.load(weights))
    elif name == 'vgg':
        model = vgg16(num_classes=1000)
        model.load_state_dict(torch.load(weights))
    else:
        raise ValueError("model name must be 'resnet' or 'alexnet' or 'vgg'.now model name is {}".format(name))
    return model

def init_image(image_path:str='./tulip.jpg',transformer_type='resnet'):
    assert os.path.exists(image_path), "image path: {} is not exists".format(image_path)
    # load image
    img = Image.open(image_path)
    # [C,H,W]
    transform = init_transform(transformer_type)
    img = transform(img)
    # [N,C,H,W]
    img = torch.unsqueeze(img,dim=0)
    return img

def main(model, img, every_channel=True):
    model = model.eval()
    output = model(img)
    for feature_map in output:
        # [N, C, H, W] -> [C, H, W]
        im = np.squeeze(feature_map.detach().numpy())  # .detach()：复制一份参数，且不进行反向传播
        # [C, H, W] -> [H, W, C]
        im = np.transpose(im, [1, 2, 0])

        plt.figure()
        show_num = 12
        row = 4
        nw = min(im.shape[2], show_num)
        if every_channel:
            for i in range(nw):
                ax = plt.subplot(row, nw//row+1, i+1)
                # [H, W, C]
                plt.imshow(im[:, :, i], cmap='gray')  #, cmap='gray'

        else:
            plt.imshow(np.mean(im, axis=2), cmap='gray')
        plt.show()

if __name__ == '__main__':
    # 可视化分析网络中的某层特征图
    model = init_model(name='alexnet', weights='./weights/alexnet.pth')
    img = init_image(image_path='./test_img/sunflower.jpg',transformer_type='alexnet')
    main(model=model, img=img)




