# few shot segmentation

## 预训练模型下载
**Pretrained model:** [ResNet-50](https://drive.google.com/file/d/11yONyypvBEYZEh9NIOJBGMdiLLAgsMgj/view?usp=sharing) | [ResNet-101](https://drive.google.com/file/d/1mX1yYvkcyOkAVjZZSIf6uMBPlooZCmpk/view?usp=sharing)

将下载好的模型放在pretrained/文件夹下。

# train
您需要在config/train.yaml中修改图片路径以及mask路径
```commandline
python train.py --device 0 --size 473 --shot 1 --fold 0
```

# test
您需要在config/test.yaml修改中修改图片路径以及mask路径
```commandline
python test.py --weights xxx.pth --num_class 20
```

# predict
为了统一测试coco,pascal,fss等小样本分割数据集，写了专门的数据读取接口。数据的划分在dataset/data/split下所示

如果你需要测试上述数据集，执行一下指令
```commandline
python predict.py --weights xxx.pth --data_path xxx --flod 0 --shot 1 --num_class 20 --vis
```

如果你需要测试你自己的数据集，需要fewshot.py中的DatasetFewShot类中进行如下修改
```commandline
self.img_path = 'xxx'
self.mask_path = 'xxx'
```

然后 执行以下指令
```commandline
python predict.py --weights xxx.pth --data_path xxx --flod 0 --shot 1 --num_class 20 --vis
```
