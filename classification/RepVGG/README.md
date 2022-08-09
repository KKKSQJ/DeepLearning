# RepVgg

- [论文解读](https://zhuanlan.zhihu.com/p/551218736)

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
python -m torch.distributed.launch --nproc_per_node=2 train.py --device 0,3
```

# test 测模型精度

```
python test.py --data xx mode xx --classes xx --weights xx --arch 
```

- data:val.txt的路径，比如data/val.txt,那么data=data.
- mode:train or depoly
- classes:类别数量
- weights:权重路径
- 注意：如果你想要测试deploy，那么你需要通过convert.py将训练生成的模型转换为推理模型

# predict 推理跑图片

```
python predict.py --arch xx --weights xx --mode xx --source xx --view-img xx --class_indices xx
```

- arch：模型结构
- weights:模型权重路径
- mode:train or deploy
- source:测试图片路径，可以是文件夹，可以是单张图片
- view_img:True #可视化图片
- --class_indices：类别名字以及类别索引的json文件
- 注意：如果mode=deploy，你需要将训练生成的模型通过convert.py转换为推理结构的模型

# convert

- convert.py:将训练生成的模型转换为推理结构的模型