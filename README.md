# DeepLearning

## 介绍
个人学习项目，用于复现论文代码，深入理解算法原理。

注意：每个项目都可独立运行。若要运行某个项目，你需要将该项目作为根目录，以便找到对应模块。

每个项目都有对应论文解读，解读详情搜 [知乎] - 琪小钧

## classification
- [x] mnist [paper read](https://zhuanlan.zhihu.com/p/459616884) / [code](https://github.com/KKKSQJ/DeepLearning/tree/master/classification/mnist)
- [x] vggNet [paper read](https://zhuanlan.zhihu.com/p/460777014) / [code](https://github.com/KKKSQJ/DeepLearning/tree/master/classification/vggNet)
- [x] resnet [paper read](https://zhuanlan.zhihu.com/p/462190341) / [code](https://github.com/KKKSQJ/DeepLearning/tree/master/classification/resnet)
- [x] coatNet [paper read](https://zhuanlan.zhihu.com/p/463033740) / [code](https://github.com/KKKSQJ/DeepLearning/tree/master/classification/coatNet)
- [x] convNext [paper read](https://zhuanlan.zhihu.com/p/473657956) / [code](https://github.com/KKKSQJ/DeepLearning/tree/master/classification/convNext)
- [x] seNet [paper read](https://zhuanlan.zhihu.com/p/479992312) / [code](https://github.com/KKKSQJ/DeepLearning/tree/master/classification/seNet)
- [x] TransFG [paper read](https://zhuanlan.zhihu.com/p/519173447) / [code](https://github.com/KKKSQJ/DeepLearning/tree/master/classification/TransFG)
- [x] RepVGG [paper read](https://zhuanlan.zhihu.com/p/551218736) / [code](https://github.com/KKKSQJ/DeepLearning/tree/master/classification/RepVGG)
- [x] efficientNet [code](https://github.com/KKKSQJ/DeepLearning/tree/master/classification/efficientNet)
- [x] shuffleNet [code](https://github.com/KKKSQJ/DeepLearning/tree/master/classification/ShuffleNet)
- [x] vision_transformer
- [ ] swin-transformer
- [ ] inception
- [ ] denseNet
- [ ] googleNet
- [ ] CBAM
- [ ] mobileNet
- [ ] resNeXt
- [ ] Xception
- [ ] SqueezeNet


## detection
- [x] FPN(实现resnet50 + fpn) [paper read](https://zhuanlan.zhihu.com/p/543486836) [code](https://github.com/KKKSQJ/DeepLearning/tree/master/detection/FPN) 
- [x] Faster-rcnn [paper read](https://zhuanlan.zhihu.com/p/543486836) [code](https://github.com/KKKSQJ/DeepLearning/tree/master/detection/fasterRcnn)
- [x] yolov7 [paper read](https://zhuanlan.zhihu.com/p/547044250) [code](https://github.com/WongKinYiu/yolov7)
- [x] RetinaNet (包含focal_loss) [code](https://github.com/KKKSQJ/DeepLearning/tree/master/detection/RetinaNet)
- [x] YOLOV5 V5.0 (实现注释，更新pt->onnx代码) [code](https://github.com/KKKSQJ/DeepLearning/tree/master/detection/yolov5)
- [x] yolox (修改了voc数据读取方式) [code](https://github.com/KKKSQJ/DeepLearning/tree/master/detection/YOLOX)
- [x] FCOS
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
- [x] U-Net [paper read](https://zhuanlan.zhihu.com/p/485647940) / [code](https://github.com/KKKSQJ/DeepLearning/tree/master/Image_segmentation/U-Net)
- [x] HR-Net-Seg [paper read](https://zhuanlan.zhihu.com/p/501094171) / [code](https://github.com/KKKSQJ/DeepLearning/tree/master/Image_segmentation/HR-Net-Seg)
- [x] DeepLabv3 [paper read](https://zhuanlan.zhihu.com/p/513233049) / [code](https://github.com/KKKSQJ/DeepLearning/tree/master/Image_segmentation/DeepLabV3)
- [x] DeepLabv3Plus [code](https://github.com/KKKSQJ/DeepLearning/tree/master/Image_segmentation/DeepLabV3Plus)
- [x] few_shot segmentation [code](https://github.com/KKKSQJ/DeepLearning/tree/master/Image_segmentation/few_shot_segmentation)
- [x] FCN [code](https://github.com/KKKSQJ/DeepLearning/tree/master/Image_segmentation/FCN)
- [ ] Mask-rcnn
- [ ] Cascade-rcnn
- [ ] UNet++
- [ ] PSPNet
- [ ] Segmenter

## metric_learning
- [x] BDB (用于图像检索) [code](https://github.com/KKKSQJ/DeepLearning/tree/master/metric_learning/BDB)
- [x] Happy-Whale (鲸鱼竞赛检索baseline) [code](https://github.com/KKKSQJ/DeepLearning/tree/master/metric_learning/Happy-Whale)


## self-supervised
- [x] MAE (实现VIT+MAE) [code](https://github.com/KKKSQJ/DeepLearning/tree/master/self-supervised/MAE)
- [x] SupCon (实现自对比学习+t-SNE可视化+swa) [code](https://github.com/KKKSQJ/DeepLearning/tree/master/self-supervised/SupCon)
- [ ] MoCo
- [ ] SwAV

## deep_stereo
- [x] Real_time_self_adaptive_deep_stereo (实时双目里立体匹配，细节待完善) [code](https://github.com/KKKSQJ/DeepLearning/tree/master/deep_stereo/Real_time_self_adaptive_depp_stereo)


## other
- [x] label_convert (三种不同标注文件之间的转换以及box可视化) [paper read](https://zhuanlan.zhihu.com/p/461488682) / [code](https://github.com/KKKSQJ/DeepLearning/tree/master/others/label_convert) 
- [x] normalization (BN、LN、IN、GN、SN图解) [paper read](https://zhuanlan.zhihu.com/p/524829507) / [code](https://github.com/KKKSQJ/DeepLearning/tree/master/others/normalization) 
- [x] DDP (模型分布式计算) [paper read](https://zhuanlan.zhihu.com/p/550554697) / [code](https://github.com/KKKSQJ/DeepLearning/tree/master/others/train_with_DDP) 
- [x] tensorboard test (可视化网络，图片，训练过程以及卷积核) / [code](https://github.com/KKKSQJ/DeepLearning/tree/master/others/tensorboard_test)
- [x] load weights test (权重部分加载) / [code](https://github.com/KKKSQJ/DeepLearning/tree/master/others/load_weights_test)
- [x] visual weights map test (特征图、卷积核可视化分析) / [code](https://github.com/KKKSQJ/DeepLearning/tree/master/others/visual_weight_feature_map_test)
- [x] class_Activation_Map_Visual (可视化CNN的类激活图) / [code]()
- [x] deploy (pytorch模型转onnx，支持自定义算子 示例) / [code](https://github.com/KKKSQJ/DeepLearning/tree/master/others/deploy)