import os
import random
from collections import defaultdict

from PIL import Image
import numpy as np

from torch.utils.data import Dataset

from dataset.transform import *


class FewShot(Dataset):
    """
    FewShot generates support-query pairs in an episodic manner,
    intended for meta-training and meta-testing paradigm.
    """

    """
    args:
        img_path: root path of images 
        mask_path: root path of mask
        id_path: txt or list of train images id 
        crop_size: cropping size of training samples
        fold: validation fold, 0 or 1 or 2 or 3
        shot: number of support pairs, 1 or 2 or 3 or 4 or 5
        episode: save the model after each snapshot episodes
        mode: train or test
    """

    # datapath, fold, transform, split, shot, use_original_imgsize
    def __init__(self,
                 img_path,
                 mask_path,
                 crop_size,
                 fold,
                 shot,
                 episode,
                 num_class,
                 mode="train",
                 ):
        super(FewShot, self).__init__()
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)

        self.crop_size = crop_size
        self.mode = mode
        self.fold = fold
        self.shot = shot
        self.episode = episode
        self.num_class = num_class

        assert os.path.exists(img_path), f"image path: {img_path} does not exists!"
        assert os.path.exists(mask_path), f"mask path: {mask_path} does not exists!"
        self.img_path = img_path
        self.mask_path = mask_path

        # if isinstance(id_path, list):
        #     self.ids = id_path
        # else:
        #     assert os.path.exists(id_path) and id_path.endswith(
        #         "txt"), f"id path: {id_path} dose not exists\t id path must is a txt file"
        #     with open(id_path, 'r', encoding='utf-8') as f:
        #         self.ids = f.read().splitlines()

        self.ids = self.build_mask_id()
        interval = num_class // 4
        if self.mode == 'train':
            # base classes = all classes - novel classes
            self.classes = set(range(1, num_class + 1)) - set(range(interval * fold + 1, interval * (fold + 1) + 1))
        else:
            # novel classes
            self.classes = set(range(interval * fold + 1, interval * (fold + 1) + 1))

        self._filter_ids()

        self.cls_to_ids = self._map_cls_to_cls()

    def __getitem__(self, item):
        # query id,image,mask
        id_q = random.choice(self.ids)
        img_q = Image.open(os.path.join(self.img_path, id_q + ".jpg")).convert("RGB")
        mask_q = Image.fromarray(np.array(Image.open(os.path.join(self.mask_path, id_q + ".png"))))
        # target class
        cls = random.choice(sorted(set(np.unique(mask_q)) & self.classes))

        # support ids, images, masks
        id_s_list, img_s_list, mask_s_list = [], [], []
        while True:
            id_s = random.choice(sorted(set(self.cls_to_ids[cls]) - {id_q} - set(id_s_list)))
            img_s = Image.open(os.path.join(self.img_path, id_s + ".jpg")).convert("RGB")
            mask_s = Image.fromarray(np.array(Image.open(os.path.join(self.mask_path, id_s + ".png"))))

            # small objects in support images are filtered
            if np.sum(np.array(mask_s) == cls) < 2 * 32 * 32:
                continue

            id_s_list.append(id_s)
            img_s_list.append(img_s)
            mask_s_list.append(mask_s)

            if len(id_s_list) == self.shot:
                break

        # transforms
        if self.mode == "train":
            img_q, mask_q = resize(img_q, mask_q, self.crop_size)
            img_q, mask_q = crop(img_q, mask_q, self.crop_size)
            img_q, mask_q = hflip(img_q, mask_q)
            for k in range(self.shot):
                img_s_list[k], mask_s_list[k] = resize(img_s_list[k], mask_s_list[k], self.crop_size)
                img_s_list[k], mask_s_list[k] = crop(img_s_list[k], mask_s_list[k], self.crop_size)
                img_s_list[k], mask_s_list[k] = hflip(img_s_list[k], mask_s_list[k])

        img_q, mask_q = normalize(img_q, mask_q)
        for k in range(self.shot):
            img_s_list[k], mask_s_list[k] = normalize(img_s_list[k], mask_s_list[k])

        # filter out irrelevant classes by setting them as background
        mask_q[(mask_q != cls) & (mask_q != 255)] = 0  # other class pixel to background
        mask_q[(mask_q == cls)] = 1  # class pixel to fg

        for k in range(self.shot):
            mask_s_list[k][(mask_s_list[k] != cls) & (mask_s_list[k] != 255)] = 0
            mask_s_list[k][mask_s_list[k] == cls] = 1

        return img_s_list, mask_s_list, img_q, mask_q, cls, id_s_list, id_q

    def __len__(self):
        return self.episode

    def build_mask_id(self):
        id_list = []
        for file in os.listdir(self.mask_path):
            if file.endswith(".png"):
                id_list.append(file.split('.')[0])
        return id_list

    # remove images that do not contain any valid classes
    # and remove images whose valid objects are all small (according to PFENet)
    def _filter_ids(self):
        for i in range(len(self.ids) - 1, -1, -1):
            mask = Image.fromarray(np.array(Image.open(os.path.join(self.mask_path, self.ids[i] + '.png'))))
            classes = set(np.unique(mask)) & self.classes
            if not classes:
                del self.ids[i]
                continue

            # remove images whose valid objects are all small (according to PFENet)
            exist_large_objects = False
            for cls in classes:
                if np.sum(np.array(mask) == cls) >= 2 * 32 * 32:
                    exist_large_objects = True
                    break
            if not exist_large_objects:
                del self.ids[i]

    # map each valid class to a list of image ids
    def _map_cls_to_cls(self):
        cls_to_ids = defaultdict(list)
        for id_ in self.ids:
            mask = np.array(Image.open(os.path.join(self.mask_path, id_ + ".png")))
            valid_classes = set(np.unique(mask)) & self.classes
            for cls in valid_classes:
                cls_to_ids[cls].append(id_)
        return cls_to_ids


class DatasetFewShot(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize, nclass=None):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = nclass
        self.benchmark = 'fewshot'
        self.shot = shot
        self.episode = 1000
        self.use_original_imgsize = use_original_imgsize

        self.img_path = os.path.join(datapath, 'Images')
        self.mask_path = os.path.join(datapath, 'Masks')
        # self.img_path = r'E:\dataset\VOC2012\VOCdevkit\VOC2012\JPEGImages'
        # self.mask_path = r'E:\practice\DeepLearning\Image_segmentation\few_shot_segmentation\SegmentationClass'
        self.transform = transform

        self.ids = self.build_mask_id()
        self.class_ids = self.build_class_ids()

        self._filter_ids()

        self.cls_to_ids = self._map_cls_to_cls()

    def __getitem__(self, item):
        # query id,image,mask
        id_q = random.choice(self.ids)
        img_q = Image.open(os.path.join(self.img_path, id_q + ".jpg")).convert("RGB")
        mask_q = Image.fromarray(np.array(Image.open(os.path.join(self.mask_path, id_q + ".png"))))
        # target class
        cls = random.choice(sorted(set(np.unique(mask_q)) & set(self.class_ids)))

        # support ids, images, masks
        id_s_list, img_s_list, mask_s_list = [], [], []
        while True:
            id_s = random.choice(sorted(set(self.cls_to_ids[cls]) - {id_q} - set(id_s_list)))
            img_s = Image.open(os.path.join(self.img_path, id_s + ".jpg")).convert("RGB")
            mask_s = Image.fromarray(np.array(Image.open(os.path.join(self.mask_path, id_s + ".png"))))

            # small objects in support images are filtered
            if np.sum(np.array(mask_s) == cls) < 2 * 32 * 32:
                continue

            id_s_list.append(id_s)
            img_s_list.append(img_s)
            mask_s_list.append(mask_s)

            if len(id_s_list) == self.shot:
                break

        # transforms
        if self.split == "trn":
            img_q, mask_q = resize(img_q, mask_q, self.crop_size)
            img_q, mask_q = crop(img_q, mask_q, self.crop_size)
            img_q, mask_q = hflip(img_q, mask_q)
            for k in range(self.shot):
                img_s_list[k], mask_s_list[k] = resize(img_s_list[k], mask_s_list[k], self.crop_size)
                img_s_list[k], mask_s_list[k] = crop(img_s_list[k], mask_s_list[k], self.crop_size)
                img_s_list[k], mask_s_list[k] = hflip(img_s_list[k], mask_s_list[k])

        # if not self.use_original_imgsize:
        #     img_q, mask_q = resize(img_q, mask_q, self.crop_size, ratio=(1.0, 1.0))

        img_q, mask_q = normalize(img_q, mask_q)
        for k in range(self.shot):
            # if not self.use_original_imgsize:
            #     img_s_list[k], mask_s_list[k] = resize(img_s_list[k], mask_s_list[k], self.crop_size, ratio=(1.0, 1.0))
            img_s_list[k], mask_s_list[k] = normalize(img_s_list[k], mask_s_list[k])

        # filter out irrelevant classes by setting them as background
        mask_q[(mask_q != cls) & (mask_q != 255)] = 0  # other class pixel to background
        mask_q[(mask_q == cls)] = 1  # class pixel to fg

        for k in range(self.shot):
            mask_s_list[k][(mask_s_list[k] != cls) & (mask_s_list[k] != 255)] = 0
            mask_s_list[k][mask_s_list[k] == cls] = 1

        # img_s_list = torch.stack(img_s_list)
        # mask_s_list = torch.stack(mask_s_list)

        batch = {'query_img': img_q,
                 'query_mask': mask_q,
                 'query_name': id_q,
                 # 'query_ignore_idx': 255,

                 'org_query_imsize': img_q.size()[-2:],

                 'support_imgs': img_s_list,
                 'support_masks': mask_s_list,
                 'support_names': id_s_list,
                 # 'support_ignore_idxs': 255,

                 'class_id': torch.tensor(cls,dtype=torch.int64)}

        # return img_s_list, mask_s_list, img_q, mask_q, cls, id_s_list, id_q
        return batch

    def __len__(self):
        return self.episode

    def build_mask_id(self):
        id_list = []
        for file in os.listdir(self.mask_path):
            if file.endswith(".png"):
                id_list.append(file.split('.')[0])
        return id_list

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        class_ids_val = [self.fold * nclass_trn + i + 1 for i in range(nclass_trn)]
        class_ids_trn = [x for x in range(1, self.nclass + 1) if x not in class_ids_val]

        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    # remove images that do not contain any valid classes
    # and remove images whose valid objects are all small (according to PFENet)
    def _filter_ids(self):
        for i in range(len(self.ids) - 1, -1, -1):
            mask = Image.fromarray(np.array(Image.open(os.path.join(self.mask_path, self.ids[i] + '.png'))))
            classes = set(np.unique(mask)) & set(self.class_ids)
            if not classes:
                del self.ids[i]
                continue

            # remove images whose valid objects are all small (according to PFENet)
            exist_large_objects = False
            for cls in classes:
                if np.sum(np.array(mask) == cls) >= 2 * 32 * 32:
                    exist_large_objects = True
                    break
            if not exist_large_objects:
                del self.ids[i]

    # map each valid class to a list of image ids
    def _map_cls_to_cls(self):
        cls_to_ids = defaultdict(list)
        for id_ in self.ids:
            mask = np.array(Image.open(os.path.join(self.mask_path, id_ + ".png")))
            valid_classes = set(np.unique(mask)) & set(self.class_ids)
            for cls in valid_classes:
                cls_to_ids[cls].append(id_)
        return cls_to_ids


if __name__ == '__main__':
    # path = r"E:\practice\DeepLearning\Image_segmentation\few_shot_segmentation\SegmentationClass"
    # f = open("train.txt",'w')
    # for i in os.listdir(path):
    #     f.write(i.split(".")[0]+"\n")
    # f.close()

    from pathlib import Path
    import matplotlib.pyplot as plt

    cfg = "../config/train.yaml"
    if isinstance(cfg, dict):
        config = cfg
    else:
        import yaml

        yaml_file = Path(cfg).name
        with open(cfg) as f:
            config = yaml.safe_load(f)

    img_path = config["data"]["img_path"]
    mask_path = config["data"]["mask_path"]
    crop_size = config["train"]["input_size"]
    fold = config["train"]["fold"]
    shot = config["train"]["shot"]
    episode = config["train"]["train_snapshot"]
    num_class = config["train"]["num_class"]

    dataset = FewShot(
        img_path=img_path,
        mask_path=mask_path,
        crop_size=crop_size,
        fold=fold,
        shot=shot,
        episode=episode,
        num_class=num_class,
        mode="train",
    )

    pallette = [0, 0, 0, 0, 255, 0]
    for b in dataset:
        # b = d[0]
        img_s_list, mask_s_list, img_q, mask_q, cls, id_s_list, id_q = b
        mask_q = mask_q.numpy().astype(np.uint8)
        mask_q = Image.fromarray(mask_q)
        mask_q.putpalette(pallette)
        plt.subplot(1, 2, 1)
        # 若要显示img，需要注释掉transform中normalize第50-53行
        plt.imshow(img_q)
        plt.title("image")

        plt.subplot(1, 2, 2)
        plt.imshow(mask_q)
        plt.title("mask")
        plt.ion()
        plt.pause(0.01)
        plt.waitforbuttonpress()
        plt.show()
