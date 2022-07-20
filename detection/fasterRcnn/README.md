# Faster RCNN 网络 结构 复现
- 论文题目：《Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks》
- 论文地址：https://arxiv.org/pdf/1506.01497.pdf
- 知乎解读：https://zhuanlan.zhihu.com/p/543486836

## 训练数据下载
- VOC2012 将数据集下载下来，放到data文件夹下。如data/VOC2012/VOCdevkit

## 预训练权重
- [resnet50_fpn_coco](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth)
- [mobilenet_v2](https://download.pytorch.org/models/mobilenet_v2-b0353104.pth)

将预训练权重下载下来放到项目更目录下即可。


## 训练
- 确保提前准备好数据集
- 确保提前下载好对应预训练模型权重
- 若要训练mobilenetv2+fasterrcnn，直接使用train_mobile_v2.py训练脚本
- 若要训练resnet50+fpn+fasterrcnn，直接使用train_resnet50_fpn.py训练脚本
- 要使用多GPU训练，使用python -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi_GPU.py指令,nproc_per_node参数为使用GPU数量
- 如果想指定使用哪些GPU设备可在指令前加上CUDA_VISIBLE_DEVICES=0,3(例如我只要使用设备中的第1块和第4块GPU设备)
- CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_GPU.py

## 推理
```
python predict.py  --weights [checkpoint_path] --num_classes 20 --source [test_data_path]
```

## 注意事项
- 在使用训练脚本时，注意要将--data-path(VOC_root)设置为自己存放VOCdevkit文件夹所在的根目录
- 由于带有FPN结构的Faster RCNN很吃显存，如果GPU的显存不够(如果batch_size小于8的话)建议在create_model函数中使用默认的norm_layer， 即不传递norm_layer变量，默认去使用FrozenBatchNorm2d(即不会去更新参数的bn层),使用中发现效果也很好。
- 训练过程中保存的results.txt是每个epoch在验证集上的COCO指标，前12个值是COCO指标，后面两个值是训练平均损失以及学习率
- 在使用预测脚本时，要将weights设置为你自己生成的权重路径。
- 使用validation文件时，注意确保你的验证集或者测试集中必须包含每个类别的目标，并且使用时只需要修改--num-classes、--data-path和--weights-path即可，其他代码尽量不要改动