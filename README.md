# DeepLearning

## 介绍
个人学习项目，用于复现论文代码，深入理解算法原理。

注意：每个项目都可独立运行。若要运行某个项目，你需要将该项目作为根目录，以便找到对应模块。

每个项目都有对应论文解读，解读详情搜 [知乎] - 琪小钧

## classification
- [x] vision_transformer
- [x] [mnist](https://zhuanlan.zhihu.com/p/459616884)
- [x] efficientNet
- [x] [vggNet](https://zhuanlan.zhihu.com/p/460777014)
- [x] [resnet](https://zhuanlan.zhihu.com/p/462190341)
- [x] [coatNet](https://zhuanlan.zhihu.com/p/463033740)
- [x] [convNext](https://zhuanlan.zhihu.com/p/473657956)
- [x] [seNet](https://zhuanlan.zhihu.com/p/479992312)
- [x] [TransFG](https://zhuanlan.zhihu.com/p/519173447)
- [ ] swin-transformer
- [ ] inception
- [ ] denseNet
- [ ] googleNet
- [ ] CBAM
- [ ] shuffleNet
- [ ] mobileNet
- [ ] resNeXt
- [ ] Xception
- [ ] SqueezeNet
- [ ] RepVgg

## detection
- [x] RetinaNet (包含focal_loss)
- [x] [FPN](https://zhuanlan.zhihu.com/p/543486836) (实现resnet50 + fpn)
- [x] YOLOV5 V5.0 (实现注释，更新pt->onnx代码)
- [x] yolox (修改了voc数据读取方式)
- [x] FCOS
- [x] [Faster-rcnn](https://zhuanlan.zhihu.com/p/543486836)
- [x] [yolov7](https://zhuanlan.zhihu.com/p/547044250)
- [ ] yoloF
- [ ] yoloR
- [ ] detr
- [ ] ssd
- [ ] Mask-rcnn
- [ ] Cascade-rcnn
- [ ] SPPNet
- [ ] CenterNet
- [ ] RepPoints
- [ ] OTA
- [ ] ATSS

## segmentation
- [x] FCN
- [x] [U-Net](https://zhuanlan.zhihu.com/p/485647940)
- [x] [HR-Net-Seg](https://zhuanlan.zhihu.com/p/501094171)
- [x] [DeepLabv3](https://zhuanlan.zhihu.com/p/513233049)
- [x] DeepLabv3Plus
- [x] few_shot segmentation
- [ ] Mask-rcnn
- [ ] Cascade-rcnn
- [ ] UNet++
- [ ] PSPNet
- [ ] Segmenter

## metric_learning
- [x] BDB (用于图像检索)
- [x] Happy-Whale (鲸鱼竞赛检索baseline)


## self-supervised
- [x] MAE (实现VIT+MAE)
- [x] SupCon (实现自对比学习+t-SNE可视化+swa)
- [ ] MoCo
- [ ] SwAV

## deep_stereo
- [x] Real_time_self_adaptive_deep_stereo (实时双目里立体匹配，细节待完善)


## other
- [x] tensorboard test (可视化网络，图片，训练过程以及卷积核)
- [x] load weights test (权重部分加载)
- [x] visual weights map test (特征图、卷积核可视化分析)
- [x] [label_convert](https://zhuanlan.zhihu.com/p/461488682) (三种不同标注文件之间的转换以及box可视化)
- [x] class_Activation_Map_Visual (可视化CNN的类激活图)
- [x] [normalization](https://zhuanlan.zhihu.com/p/524829507) (BN、LN、IN、GN、SN图解)