# HR-Net 网络 结构 复现
- 论文题目：《Deep High-Resolution Representation Learning for Visual Recognition》
- 论文地址：https://arxiv.org/pdf/1908.07919.pdf
- 论文讲解：https://zhuanlan.zhihu.com/p/501094171

## 训练数据下载
链接：https://pan.baidu.com/s/1sRomCSZ-Wo_qlCMMQMbkIA?pwd=9noc 
提取码：9noc

## 本文硬件配置
- linux18.04
- RTX3090

## 训练
- 配置文件：config/example.yaml

您需要在配置文件中 修改您的数据路径，详细注解见config/example.yaml

- train
```
python train.py --config_name config/example.yaml
```

## 推理
```
python predict.py --cfg config/example.yaml --weights [checkpoint_path] --source [test_data_path]
```

## tensorboard
```commandline
tensorboard --logdir=log_path
```