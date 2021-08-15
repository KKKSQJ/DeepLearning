# VIT
## 本项目复现论文《AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE》
## 预训练模型链接：
* https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth 
-下载模型并且更改模型名字为：vit_base_patch16_224_in21k.pth,置于项目根目录下即可
  
## 数据集链接
* http://download.tensorflow.org/example_images/flower_photos.tgz

## 运行
- 1.设置训练数据路径
- 2.python train.py