# 读取数据
import json
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt


def read_split_data(data_root, save_dir, val_rate=0.2, plot_iamge=False):
    # 随机种子，确保每次结果可复现
    random.seed(0)

    # 遍历data_root下的文件夹，一个文件夹对应一个类别
    classes = [cla for cla in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, cla))]
    # 排序
    classes.sort()

    # 分类任务，需要索引，所以创建类别名称以及对应的索引
    class_indices = dict((k, v) for v, k in enumerate(classes))
    # 将类别以及索引写入json文件
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open(save_dir + '/class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # 存放训练集图片路径
    train_images_path = []
    # 存放验证集图片路径
    val_images_path = []
    # 存放训练集标签
    train_labels = []
    # 存放验证集标签
    val_labels = []
    # 存放每个类别的图片数量
    every_class_num = []
    # 图片文件所能支持的格式
    supported = ['.jpg', '.jpeg', '.png']

    # 将图片路径保存至txt
    train_txt = open(save_dir + '/train.txt', 'w')
    val_txt = open(save_dir + '/val.txt', 'w')

    # 遍历每一个标签文件夹，读取图片
    for cla in tqdm(classes):
        cla_path = os.path.join(data_root, cla)
        # 遍历获取对应文件夹下的所有图片
        images = [os.path.join(cla_path, i) for i in os.listdir(cla_path) if
                  os.path.splitext(i)[-1].lower() in supported]
        # 获取图片标签
        image_class = class_indices[cla]
        # 记录该类别的图片数量
        every_class_num.append(len(images))
        # 按比例划分训练集和验证集
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for image_path in images:
            if image_path in val_path:
                val_txt.write(image_path + '\n')
                val_images_path.append(image_path)
                val_labels.append(image_class)
            else:
                train_txt.write(image_path+'\n')
                train_images_path.append(image_path)
                train_labels.append(image_class)

    train_txt.close()
    val_txt.close()

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    if plot_iamge:
        plt.bar(range(len(classes)),every_class_num,align='center')
        plt.xticks(range(len(classes)),classes)
        for i,v in enumerate(every_class_num):
            plt.text(x=i,y=v+5,s=str(v),ha='center')
        plt.xlabel('image classes')
        plt.ylabel('number of image')
        plt.title('class distribution')
        plt.savefig(os.path.join(save_dir,'classes.jpg'))
    return train_images_path,val_images_path,train_labels,val_labels,every_class_num

if __name__ == '__main__':
    read_split_data(data_root=r'E:\dataset\flow_data\train',save_dir='./',plot_iamge=True)


