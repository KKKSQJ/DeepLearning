# Swin Transformer
- 论文地址：
- 官方github：


# 环境安装
- PyTorch>=1.8.0
- torchvision>=0.9.0
- CUDA>=10.2
- timm==0.4.12
- pencv-python==4.4.0.46 
- termcolor==1.1.0 
- yacs==0.1.8

模型脚本中使用了```--fused_window_process```，需要安装 fused windwo process

```bash
cd kernels/window_process
python setup.py install #--user
```


# 训练集数据格式
- 对于标准的图像分类数据集，文件结构如下所示
```
$ tree data
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...

```

- 为了提高从海量小文件中读取图像时的速度，还支持从zipped ImageNet中读取文件，其中压缩包内应该包含四个文件：
  - ```train.zip```,```val.zip```：它存储用于训练和验证的压缩文件夹
  - ```train_map.txt```, ```val_map.txt```：它存储图片对应zip的相对位置，以及标签
```
$ tree data
data
└── ImageNet-Zip
    ├── train_map.txt
    ├── train.zip
    ├── val_map.txt
    └── val.zip

$ head -n 5 data/ImageNet-Zip/val_map.txt
ILSVRC2012_val_00000001.JPEG	65
ILSVRC2012_val_00000002.JPEG	970
ILSVRC2012_val_00000003.JPEG	230
ILSVRC2012_val_00000004.JPEG	809
ILSVRC2012_val_00000005.JPEG	516

$ head -n 5 data/ImageNet-Zip/train_map.txt
n01440764/n01440764_10026.JPEG	0
n01440764/n01440764_10027.JPEG	0
n01440764/n01440764_10029.JPEG	0
n01440764/n01440764_10040.JPEG	0
n01440764/n01440764_10042.JPEG	0
```

- 还支持从txt中读取数据。txt中存放图片的绝对路径，以及标签
```
$ tree data
data
├── train.txt
└── val.val

$ head -n 5 data/train.txt
xxx/xxx.JPEG 0
xxx/xxx.JPEG 1
xxx/xxx.JPEG 2
xxx/xxx.JPEG 0
xxx/xxx.JPEG 6
```

详细的说明见dataLoader/build文件
- 标准图像分类格式，另```dataset:imagenet```
- 从TXT中读取数据，另```dataset:read_from_txt```

# 配置文件
```config.py```存放管理配置参数。可以融合命令行以及xx.yaml参数。

# train

- 单卡训练

```
python main.py --cfg <config-file> 
--data-path <data-path> 
--dataset <dataset> 
--num-classes <num-classes> 
--batch-size <bs> 
--device 0 
--pretrained <pretrained>
--output <output_dir>
--tag <job-tag>
```

- 多卡训练 DDP

--nproc_per_node=2：使用2张GPU

--device:指令GPU训练

```
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 main.py --cfg <config-file> --data-path <data-path> --dataset <dataset> --num-classes <num-classes> --batch-size <bs> --device 0 --pretrained <pretrained> --output <output_dir> --tag <job-tag>
```

For example:
```
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 main.py --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --data-path <data-path> --dataset imagenet --num-classes 1000 --batch-size 128 --device 1,2 --pretrained swin_tiny_patch4_window7_224.pth --output output --tag tiny
```

参数说明：
- cfg：模型配置文件
- data-path：数据路径
- dataset：imagenet(标准数据格式) or read_from_txt(从txt读取数据)
- num-classes：类别数量
- batch-size：每个gpu的bs数量
- device：指定GPU训练 
- pretrained： 预训练模型 
- output：模型以及日志输出文件夹
- tag：最终输出的文件名字为output/cfg_name/tag

注意：
- 如果想要zipped Imagenet，在命令行中使用```--zip```。
- 当GPU内存不够时，可以试试以下操作：
  - 通过添加```--accumulation-steps <steps>```使用梯度累积，根据需要设置合适的<steps> 2
  - 通过添加 ```--use-checkpoint``` 来使用梯度检查点，例如，它在训练 ```Swin-B``` 时可以节省大约 60% 的内存。
- 若是断开模型训练，再一次开启训练的时候，默认从上一次的epoch继续训练。
- 更加详细的参数，见config.py以及 python main.py --help

# eval 模型精度评估

```
python main.py --eval --cfg <config-file> --data-path <data-path> --dataset <dataset> --num-classes <num-classes> --batch-size <bs> --device 0 --resume <checkpoint>
```

参数说明：
- eval：只进行模型验证
- cfg:模型配置文件
- data-path:数据路径
- dataset:imagenet or read_from_txt
- num-classes:类别数量
- batch-size：bs
- device:指定gpu
- resume：<checkpoint>推理模型路径


# predict 推理跑图片

```
python predict.py --weights <checkpoint> --cfg <cfg_file> --num-classes <classes> --source <data> --img-size <size> -v -s --class-indices <indices> --device <device>
```

- weights:模型权重路径
- cfg:模型配置文件
- num-classes: 类别数量
- source:测试图片路径，可以是文件夹，可以是单张图片
- img-size：输入图像大小
- v:可视化图片
- s:保存日志文件
- class_indices：类别名字以及类别索引的json文件
- device: 指定GPU测试

# 吞吐量
```
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py \
--cfg <config-file> --data-path <imagenet-path> --batch-size 64 --throughput --disable_amp
```

# 可供选择的训练策略
- 数据增强：
- 学习率：
- 优化器：
- 损失函数
