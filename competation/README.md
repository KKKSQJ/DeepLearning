# Competation

## 主项目主要维护用于比赛的代码，争取打造一份在比赛中具有竞争力的baseline，甚至是直接可以落地的项目代码。

## 目标检测
### yolov5
- 版本v6.0，fork https://github.com/ultralytics/yolov5
- 数据集：VOC2012
- 使用预训练模型：https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt
- 训练命令：python python train.py --data coco128.yaml --cfg yolov5s.yaml --weights yolov5s.pth --batch-size 64
- 实验1：使用yolov5默认参数配置(BCELoss+CIou),mAP:
- 实验2：focal_loss(1.5, 0.25) + CIou, mAP:
- 实验3：focal_loss(1.5, 0.25) + AlphaIoULoss(3),mAP:
- 实验4：BCELoss + AlphaIouLoss(3), mAP:
- 实验5：在最后15个epoch，关闭mosaic,默认配置，mAP:
- 实验6：使用SAM优化器，(会增加训练时间)，默认配置，mAP:
#### 模型训练总结：
#### 模型部署
- onnx
- openvivo
- tensorrt

### yolox
### cascade-rcnn-dcn
### swin-transformer

## 图像分类
### efficinet-net
### resnet
### swin-transformer
### inception

## 图像分割
### deeplab
### fcn
### HRNet
### Unet
### mask-rcnn

### 图像检索/reid
### 
