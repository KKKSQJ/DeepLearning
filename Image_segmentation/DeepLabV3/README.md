# DeepLabV3 网络 结构 复现
- 论文题目：《Rethinking Atrous Convolution for Semantic Image Segmentation》
- 论文地址：https://arxiv.org/abs/1706.05587

## 训练数据下载
链接：https://pan.baidu.com/s/1sRomCSZ-Wo_qlCMMQMbkIA?pwd=9noc 
提取码：9noc

## 预训练权重
- [deeplabv3_resnet50_coco](https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth)
- [deeplabv3_resnet101_coco](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth)
- [depv3_mobilenetv3_large_coco](https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large-fc3c493d.pth)

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