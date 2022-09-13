# ShuffleNet


# 训练集数据格式

```
--data_dir
   |--label1
        |--xx.jpg
   |--label2
        |--xx.jpg
```

# train

- 单卡训练 参数详见config/train.yaml,同时你也可以通过命令行来覆盖掉train.yaml的参数

```
python train.py --arch xx --data_path xx --classes xx --device 0 
```

- 多卡训练 DDP

--nproc_per_node=2：使用2张GPU

--device:指令GPU训练

```
python -m torch.distributed.launch --nproc_per_node=2 --master_port 29501 train.py --device 0,5 --batch_size 200
```

# test 测模型精度

```
python test.py --data xx mode xx --classes xx --weights xx --arch 
```

- data:val.txt的路径，比如data/val.txt,那么data=data.
- mode:train or depoly
- classes:类别数量
- weights:权重路径


# predict 推理跑图片

```
python predict.py --arch xx --weights xx --source xx --view-img --class_indices xx
```

- arch：模型结构
- weights:模型权重路径
- source:测试图片路径，可以是文件夹，可以是单张图片
- view_img #可视化图片
- --class_indices：类别名字以及类别索引的json文件

