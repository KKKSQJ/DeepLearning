import json
import random

import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url, list_dir, check_integrity, extract_archive, verify_str_arg
from torchvision.datasets.folder import default_loader
from torchvision.datasets import VisionDataset
from tqdm import tqdm

from dataLoader.base_dataset import BasicDataset

import os
import scipy
from scipy import io
import scipy.misc
import pandas as pd
import numpy as np


class CUB(BasicDataset):
    def __init__(self,
                 root,
                 data_len=None,
                 train=True,
                 transforms=None):
        assert os.path.exists(root), f"root: {root} does not exists!"
        image_root = os.path.join(root, "images")

        img_name_list = []
        with open(os.path.join(root, "images.txt")) as f:
            for line in f.readlines():
                img_name_list.append(line.strip().split(" ")[-1])

        labels = []
        with open(os.path.join(root, "image_class_labels.txt")) as f:
            for line in f.readlines():
                labels.append(int(line.strip().split(" ")[-1]) - 1)  # label从0开始

        train_test = []
        with open(os.path.join(root, "train_test_split.txt")) as f:
            for line in f.readlines():
                train_test.append(int(line.strip().split(" ")[-1]))

        json_file = open('class_indices.json', 'w', encoding='utf-8')
        data = {}
        with open(os.path.join(root, "classes.txt")) as f:
            for line in f.readlines():
                line = line.strip().split(" ")
                data[str(int(line[0]) - 1)] = line[1]
        json_str = json.dumps(data, indent=4)
        json_file.write(json_str)
        json_file.close()

        train_file_list = [x for i, x in zip(train_test, img_name_list) if i]
        val_file_list = [x for i, x in zip(train_test, img_name_list) if not i]

        if train:
            images = [os.path.join(image_root, x) for x in train_file_list[:data_len]]
            labels = [x for i, x in zip(train_test, labels) if i][:data_len]
            ids = [x for i, x in zip(train_test, img_name_list) if i][:data_len]
        if not train:
            images = [os.path.join(image_root, x) for x in val_file_list[:data_len]]
            labels = [x for i, x in zip(train_test, labels) if not i][:data_len]
            ids = [x for i, x in zip(train_test, img_name_list) if not i][:data_len]

        super(CUB, self).__init__(ids, images, labels, transforms)


class Cars(BasicDataset):
    def __init__(self,
                 mat_anno,
                 data_dir,
                 car_names,
                 data_len=None,
                 cleaned=None,
                 transforms=None):
        """
        Args:
            mat_anno (string): Path to the MATLAB annotation file.
            data_dir (string): Directory with all the images.
            transforms (callable, optional): Optional transform to be applied
                on a sample.
        """
        full_data_set = io.loadmat(mat_anno)
        car_annotations = full_data_set['annotations']
        car_annotations = car_annotations[0]

        if cleaned is not None:
            cleaned_annos = []
            print("Cleaning up data set (only take pics with rgb chans)...")
            clean_files = np.loadtxt(cleaned, dtype=str)

            for c in car_annotations:
                if c[-1][0] in clean_files:
                    cleaned_annos.append(c)
            car_annotations = cleaned_annos

        self.car_names = scipy.io.loadmat(car_names)['class_names']
        self.car_names = np.array(self.car_names[0])

        images = [os.path.join(data_dir, x[-1][0]) for x in car_annotations][:data_len]
        labels = [torch.from_numpy(np.array(x[-2][0][0].astype(np.float32))).long() - 1 for x in car_annotations][
                 :data_len]
        ids = car_annotations[:data_len]

        super(Cars, self).__init__(ids, images, labels, transforms)

    def map_class(self, id):
        id = np.ravel(id)
        ret = self.car_names[id - 1][0][0]
        return ret


class dogs(BasicDataset):
    """`Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        cropped (bool, optional): If true, the images will be cropped into the bounding box specified
            in the annotations
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset tar files from the internet and
            puts it in root directory. If the tar files are already downloaded, they are not
            downloaded again.
    """
    folder = 'dog'
    download_url_prefix = 'http://vision.stanford.edu/aditya86/ImageNetDogs'

    def __init__(self,
                 root,
                 train=True,
                 cropped=False,
                 data_len=None,
                 transforms=None,
                 target_transforms=None,
                 download=False):
        self.root = root
        self.train = train
        self.cropped = cropped
        self.data_len = data_len
        self.transforms = transforms
        self.target_transforms = target_transforms

        if download:
            self.download()

        split = self.load_split()

        self.images_folder = os.path.join(self.root, 'Images')
        self.annotations_folder = os.path.join(self.root, 'Annotation')
        self._breeds = list_dir(self.images_folder)

        if self.cropped:
            self._breed_annotations = [[(annotation, box, idx)
                                        for box in self.get_boxes(os.path.join(self.annotations_folder, annotation))]
                                       for annotation, idx in split]
            self._flat_breed_annotations = sum(self._breed_annotations, [])

            self._flat_breed_images = [(annotation + '.jpg', idx) for annotation, box, idx in
                                       self._flat_breed_annotations][:data_len]
        else:
            self._breed_images = [(annotation + '.jpg', idx) for annotation, idx in split]

            self._flat_breed_images = self._breed_images[:data_len]

            self.classes = ["Chihuaha",
                            "Japanese Spaniel",
                            "Maltese Dog",
                            "Pekinese",
                            "Shih-Tzu",
                            "Blenheim Spaniel",
                            "Papillon",
                            "Toy Terrier",
                            "Rhodesian Ridgeback",
                            "Afghan Hound",
                            "Basset Hound",
                            "Beagle",
                            "Bloodhound",
                            "Bluetick",
                            "Black-and-tan Coonhound",
                            "Walker Hound",
                            "English Foxhound",
                            "Redbone",
                            "Borzoi",
                            "Irish Wolfhound",
                            "Italian Greyhound",
                            "Whippet",
                            "Ibizian Hound",
                            "Norwegian Elkhound",
                            "Otterhound",
                            "Saluki",
                            "Scottish Deerhound",
                            "Weimaraner",
                            "Staffordshire Bullterrier",
                            "American Staffordshire Terrier",
                            "Bedlington Terrier",
                            "Border Terrier",
                            "Kerry Blue Terrier",
                            "Irish Terrier",
                            "Norfolk Terrier",
                            "Norwich Terrier",
                            "Yorkshire Terrier",
                            "Wirehaired Fox Terrier",
                            "Lakeland Terrier",
                            "Sealyham Terrier",
                            "Airedale",
                            "Cairn",
                            "Australian Terrier",
                            "Dandi Dinmont",
                            "Boston Bull",
                            "Miniature Schnauzer",
                            "Giant Schnauzer",
                            "Standard Schnauzer",
                            "Scotch Terrier",
                            "Tibetan Terrier",
                            "Silky Terrier",
                            "Soft-coated Wheaten Terrier",
                            "West Highland White Terrier",
                            "Lhasa",
                            "Flat-coated Retriever",
                            "Curly-coater Retriever",
                            "Golden Retriever",
                            "Labrador Retriever",
                            "Chesapeake Bay Retriever",
                            "German Short-haired Pointer",
                            "Vizsla",
                            "English Setter",
                            "Irish Setter",
                            "Gordon Setter",
                            "Brittany",
                            "Clumber",
                            "English Springer Spaniel",
                            "Welsh Springer Spaniel",
                            "Cocker Spaniel",
                            "Sussex Spaniel",
                            "Irish Water Spaniel",
                            "Kuvasz",
                            "Schipperke",
                            "Groenendael",
                            "Malinois",
                            "Briard",
                            "Kelpie",
                            "Komondor",
                            "Old English Sheepdog",
                            "Shetland Sheepdog",
                            "Collie",
                            "Border Collie",
                            "Bouvier des Flandres",
                            "Rottweiler",
                            "German Shepard",
                            "Doberman",
                            "Miniature Pinscher",
                            "Greater Swiss Mountain Dog",
                            "Bernese Mountain Dog",
                            "Appenzeller",
                            "EntleBucher",
                            "Boxer",
                            "Bull Mastiff",
                            "Tibetan Mastiff",
                            "French Bulldog",
                            "Great Dane",
                            "Saint Bernard",
                            "Eskimo Dog",
                            "Malamute",
                            "Siberian Husky",
                            "Affenpinscher",
                            "Basenji",
                            "Pug",
                            "Leonberg",
                            "Newfoundland",
                            "Great Pyrenees",
                            "Samoyed",
                            "Pomeranian",
                            "Chow",
                            "Keeshond",
                            "Brabancon Griffon",
                            "Pembroke",
                            "Cardigan",
                            "Toy Poodle",
                            "Miniature Poodle",
                            "Standard Poodle",
                            "Mexican Hairless",
                            "Dingo",
                            "Dhole",
                            "African Hunting Dog"]

            super(dogs, self).__init__()

    def __len__(self):
        return len(self._flat_breed_images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, target_class = self._flat_breed_images[index]
        image_path = os.path.join(self.images_folder, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.cropped:
            image = image.crop(self._flat_breed_annotations[index][1])

        if self.transforms is not None:
            image = self.transforms(image)

        if self.target_transforms is not None:
            target_class = self.target_transforms(target_class)

        return image, target_class

    def download(self):
        import tarfile

        if os.path.exists(os.path.join(self.root, 'Images')) and os.path.exists(os.path.join(self.root, 'Annotation')):
            if len(os.listdir(os.path.join(self.root, 'Images'))) == len(
                    os.listdir(os.path.join(self.root, 'Annotation'))) == 120:
                print('Files already downloaded and verified')
                return

        for filename in ['images', 'annotation', 'lists']:
            tar_filename = filename + '.tar'
            url = self.download_url_prefix + '/' + tar_filename
            download_url(url, self.root, tar_filename, None)
            print('Extracting downloaded file: ' + os.path.join(self.root, tar_filename))
            with tarfile.open(os.path.join(self.root, tar_filename), 'r') as tar_file:
                tar_file.extractall(self.root)
            os.remove(os.path.join(self.root, tar_filename))

    def load_split(self):
        if self.train:
            split = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(os.path.join(self.root, 'train_list.mat'))['labels']
        else:
            split = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(os.path.join(self.root, 'test_list.mat'))['labels']

        split = [item[0][0] for item in split]
        labels = [item[0] - 1 for item in labels]
        return list(zip(split, labels))

    @staticmethod
    def get_boxes(path):
        import xml.etree.ElementTree
        e = xml.etree.ElementTree.parse(path).getroot()
        boxes = []
        for objs in e.iter('object'):
            boxes.append([int(objs.find('bndbox').find('xmin').text),
                          int(objs.find('bndbox').find('ymin').text),
                          int(objs.find('bndbox').find('xmax').text),
                          int(objs.find('bndbox').find('ymax').text)])
        return boxes


class NASBirds(Dataset):
    """`NABirds <https://dl.allaboutbirds.org/nabirds>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            train (bool, optional): If True, creates dataset from training set, otherwise
               creates from test set.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    base_folder = 'nabirds/images'

    def __init__(self, root, train=True, transform=None):
        super().__init__()
        dataset_path = os.path.join(root, 'nabirds')
        self.root = root
        self.loader = default_loader
        self.train = train
        self.transform = transform

        image_paths = pd.read_csv(os.path.join(dataset_path, 'images.txt'),
                                  sep=' ', names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(dataset_path, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        # Since the raw labels are non-continuous, map them to new ones
        self.label_map = get_continuous_class_map(image_class_labels['target'])
        train_test_split = pd.read_csv(os.path.join(dataset_path, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        data = image_paths.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        # Load in the train / test split
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        # Load in the class data
        self.class_names = load_class_names(dataset_path)
        self.class_hierarchy = load_hierarchy(dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = self.label_map[sample.target]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        return img, target


def get_continuous_class_map(class_labels):
    label_set = set(class_labels)
    return {k: i for i, k in enumerate(label_set)}


def load_class_names(dataset_path=''):
    names = {}

    with open(os.path.join(dataset_path, 'classes.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            class_id = pieces[0]
            names[class_id] = ' '.join(pieces[1:])

    return names


def load_hierarchy(dataset_path=''):
    parents = {}

    with open(os.path.join(dataset_path, 'hierarchy.txt')) as f:
        for line in f:
            pieces = line.strip().split()
            child_id, parent_id = pieces
            parents[child_id] = parent_id

    return parents


class INat2017(VisionDataset):
    """`iNaturalist 2017 <https://github.com/visipedia/inat_comp/blob/master/2017/README.md>`_ Dataset.
        Args:
            root (string): Root directory of the dataset.
            split (string, optional): The dataset split, supports ``train``, or ``val``.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'train_val_images/'
    file_list = {
        'imgs': ('https://storage.googleapis.com/asia_inat_data/train_val/train_val_images.tar.gz',
                 'train_val_images.tar.gz',
                 '7c784ea5e424efaec655bd392f87301f'),
        'annos': ('https://storage.googleapis.com/asia_inat_data/train_val/train_val2017.zip',
                  'train_val2017.zip',
                  '444c835f6459867ad69fcb36478786e7')
    }

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(INat2017, self).__init__(root, transform=transform, target_transform=target_transform)
        self.loader = default_loader
        self.split = verify_str_arg(split, "split", ("train", "val",))

        if self._check_exists():
            print('Files already downloaded and verified.')
        elif download:
            if not (os.path.exists(os.path.join(self.root, self.file_list['imgs'][1]))
                    and os.path.exists(os.path.join(self.root, self.file_list['annos'][1]))):
                print('Downloading...')
                self._download()
            print('Extracting...')
            extract_archive(os.path.join(self.root, self.file_list['imgs'][1]))
            extract_archive(os.path.join(self.root, self.file_list['annos'][1]))
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')
        anno_filename = split + '2017.json'
        with open(os.path.join(self.root, anno_filename), 'r') as fp:
            all_annos = json.load(fp)

        self.annos = all_annos['annotations']
        self.images = all_annos['images']

    def __getitem__(self, index):
        path = os.path.join(self.root, self.images[index]['file_name'])
        target = self.annos[index]['category_id']

        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self):
        return len(self.images)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.base_folder))

    def _download(self):
        for url, filename, md5 in self.file_list.values():
            download_url(url, root=self.root, filename=filename)
            if not check_integrity(os.path.join(self.root, filename), md5):
                raise RuntimeError("File not found or corrupted.")


class MyDataset(BasicDataset):
    def __init__(self, root, data_len=None, train=True, transform=None):
        random.seed(0)  # 保证随机结果可复现
        assert os.path.exists(root), "data path:{} does not exists".format(root)

        # 遍历文件夹，一个文件夹对应一个类别
        classes = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
        # 排序，保证顺序一致
        classes.sort()

        # 生成类别名称以及对应的数字索引
        class_indices = dict((k, v) for v, k in enumerate(classes))
        json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json_str)

        train_images_path = []  # 存储训练集的所有图片路径
        train_images_label = []  # 存储训练集图片对应索引信息
        val_images_path = []  # 存储验证集的所有图片路径
        val_images_label = []  # 存储验证集图片对应索引信息
        every_class_num = []  # 存储每个类别的样本总数
        supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型

        # 遍历每个文件夹下的文件
        for cla in tqdm(classes):
            cla_path = os.path.join(root, cla)
            # 遍历获取supported支持的所有文件路径
            images = [os.path.join(root, cla, i) for i in os.listdir(cla_path) if
                      os.path.splitext(i)[-1] in supported]
            # 获取该类别对应的索引
            image_class = class_indices[cla]
            # 记录该类别的样本数量
            every_class_num.append(len(images))
            # 按比例随机采样验证样本
            val_path = random.sample(images, k=int(len(images) * 0.2))

            train_txt = open('train.txt', 'w')
            val_txt = open('val.txt', 'w')
            for img_path in images:
                if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                    val_txt.write(img_path + "\n")
                    val_images_path.append(img_path)
                    val_images_label.append(image_class)
                else:  # 否则存入训练集
                    train_txt.write(img_path + "\n")
                    train_images_path.append(img_path)
                    train_images_label.append(image_class)
            train_txt.close()
            val_txt.close()

        if train:
            _ids = train_images_path[:data_len]
            _images = train_images_path[:data_len]
            _labels = train_images_label[:data_len]
        else:
            _ids = val_images_path[:data_len]
            _images = val_images_path[:data_len]
            _labels = val_images_label[:data_len]



        print("{} images were found in the dataset.".format(sum(every_class_num)))
        print("{} images for training.".format(len(train_images_path)))
        print("{} images for validation.".format(len(val_images_path)))
        super(MyDataset, self).__init__(ids=_ids, images=_images, labels=_labels, transforms=transform)


if __name__ == '__main__':
    # dataset = CUB(root=r'E:\dataset\classifition\CUB_200_2011\CUB_200_2011')
    # d = dataset[0]
    # print(d)

    import matplotlib.pyplot as plt
    from dataLoader.autoaugment import AutoAugImageNetPolicy
    from torchvision import transforms
    from PIL import Image

    transform = transforms.Compose(
        [
            transforms.Resize((600, 600), Image.BILINEAR),
            transforms.CenterCrop((448, 448)),
            AutoAugImageNetPolicy()
        ]
    )

    dataset = CUB(root=r'E:\dataset\classifition\CUB_200_2011\CUB_200_2011', data_len=None, train=True,
                  transforms=transform)

    # dataset = MyDataset(root=r'E:\dataset\classifition\flow_data\train', data_len=None, train=True,
    #                     transform=transform)

    for index, data in enumerate(dataset):
        image = data[0]
        label = data[1]
        plt.imshow(image)
        plt.title("label:" + str(label))
        print("size:" + str(image.size))
        # plt.text(0,0-10,image.size)
        plt.ion()
        plt.pause(0.01)
        plt.waitforbuttonpress()
        # plt.close()
        plt.show()
