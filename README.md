# DeepLearning

## 介绍
个人学习项目，用于复现论文代码，深入理解算法原理。

注意：每个项目都可独立运行。若要运行某个项目，你需要将该项目作为根目录，以便找到对应模块。

每个项目都有对应论文解读，解读详情搜 [知乎] - 琪小钧

## classification
- vision_transformer(完成)
- mnist 手写字符(完成)
- efficientNet(完成)
- vggNet(完成)
- resnet(完成)
- coatNet(完成)
- convNext(完成)
- seNet(完成)

## detection
- RetinaNet(完成) 包含focal_loss
- FPN(半完成) 实现resnet50 + fpn
- YOLOV5 V5.0(完成) 实现注释，更新pt->onnx代码
- yolox(完成) 修改了voc数据读取方式
- FCOS (完成)

## segmentation
- FCN(完成)
- U-Net(完成)
- HR-Net-Seg(完成)

## metric_learning
- BDB(完成) 用于图像检索
- Happy-Whale(完成) 鲸鱼竞赛检索baseline


## self-supervised
- MAE(完成) 实现VIT+MAE
- SupCon(完成) 实现自对比学习+t-SNE可视化+swa

## deep_stereo
- Real_time_self_adaptive_deep_stereo(实时双目里立体匹配，细节待完善)


## other
- tensorboard test(完成) 可视化网络，图片，训练过程以及卷积核
- load weights test(完成) 权重部分加载
- visual weights map test(完成) 特征图、卷积核可视化分析
- label_convert(完成) 三种不同标注文件之间的转换以及box可视化
- class_Activation_Map_Visual(完成) 可视化CNN的类激活图