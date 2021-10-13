# encoding: utf-8
from PIL import Image
from torchvision import transforms as T
from utils.data_aug import RandomErasing, Cutout
import random


class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
        height (int): target height.
        width (int): target width.
        p (float): probability of performing this transformation. Default: 0.5.
    """

    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if random.random() < self.p:
            return img.resize((self.width, self.height), self.interpolation)
        new_width, new_height = int(
            round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop(
            (x1, y1, x1 + self.width, y1 + self.height))
        return croped_img

def pad_shorter(x):
    h,w = x.size[-2:]
    s = max(h, w)
    new_im = Image.new("RGB", (s, s))
    new_im.paste(x, ((s-h)//2, (s-w)//2))
    return new_im

class TrainTransform(object):
    def __init__(self, data):
        self.data = data

    def __call__(self, x):
        if self.data == 'person':
            x = T.Resize((384, 128))(x)
        elif self.data == 'car':
            x = pad_shorter(x)
            x = T.Resize((256, 256))(x)
            x = T.RandomCrop((224, 224))(x)
        elif self.data == 'cub':
            x = pad_shorter(x)
            x = T.Resize((256, 256))(x)
            x = T.RandomCrop((224, 224))(x)
        elif self.data == 'clothes':
            x = pad_shorter(x)
            x = T.Resize((256, 256))(x)
            x = T.RandomCrop((224, 224))(x)
        elif self.data == 'product':
            x = pad_shorter(x)
            x = T.Resize((256, 256))(x)
            x = T.RandomCrop((224, 224))(x)
        elif self.data == 'cifar':
            x = T.Resize((40, 40))(x)
            x = T.RandomCrop((32, 32))(x)
        elif self.data == 'button':
            x = pad_shorter(x)
            x = T.Resize((256, 256))(x)
            x = T.RandomCrop((224, 224))(x)
        x = T.RandomHorizontalFlip()(x)
        x = T.ToTensor()(x)
        x = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)

        if self.data == 'person':
            x = Cutout(probability = 0.5, size=64, mean=[0.0, 0.0, 0.0])(x)
        else:
            x = RandomErasing(probability = 0.5, mean=[0.0, 0.0, 0.0])(x)
        return x


class TestTransform(object):
    def __init__(self, data, flip=False):
        self.data = data
        self.flip = flip

    def __call__(self, x=None):
        if self.data == 'cub':
            x = pad_shorter(x)
            x = T.Resize((256, 256))(x)
        elif self.data == 'car':
            #x = pad_shorter(x)
            x = T.Resize((256, 256))(x)
        elif self.data == 'clothes':
            x = pad_shorter(x)
            x = T.Resize((256, 256))(x)
        elif self.data == 'product':
            x = pad_shorter(x)
            x = T.Resize((224, 224))(x)
        elif self.data == 'person':
            x = T.Resize((384, 128))(x)
        elif self.data == 'button':
            x = pad_shorter(x)
            x = T.Resize((224, 224))(x)

        if self.flip:
            x = T.functional.hflip(x)
        x = T.ToTensor()(x)
        x = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])(x)
        return x
