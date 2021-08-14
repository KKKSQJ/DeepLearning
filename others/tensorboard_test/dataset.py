from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset

class DataSet(Dataset):
    def __init__(self, data:dict, transform=None):
        self.transform = transform
        self.images_path = []
        self.images_label = []

        for key, value in data.items():
            self.images_path.extend(value)
            self.images_label.extend([int(key)]*len(value))
        assert len(self.images_path) == len(self.images_label)

        delete_img = []
        for index, img_path in tqdm(enumerate(self.images_path)):
            img = Image.open(img_path)
            w, h = img.size
            ratio = w / h
            if ratio > 10 or ratio < 0.1:
                delete_img.append(index)
                # print(img_path, ratio)

        for index in delete_img[::-1]:
            self.images_path.pop(index)
            self.images_label.pop(index)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        # img = img.convert("RGB")
        label = self.images_label[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


if __name__ == '__main__':
    a ={
        "0":[1,2,3,4],
        "1":[4,8,6,]
    }
    images_path  = []
    images_label = []
    for key, value in a.items():
        images_path.extend(value)
        images_label.extend([int(key)] * len(value))
    print(1)
