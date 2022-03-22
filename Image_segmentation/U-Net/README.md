# U-Net

## Data
- 训练数据可以从kaggle下载：https://www.kaggle.com/datasets/ipythonx/carvana-image-masking-png
- 输入图像和目标mask应该分别放在data/imgs和data/mask。(注意：imgs和masks文件夹不可以包含其他字文件夹或者其他文件)
- 如果你的数据存放在其他地方，你可以使用一下命令改变你的数据集路径
```
python train.py --imgs_dir [imgs_path] --mask_dir [mask_path]
```
- 对于训练数据，images应该是RGB格式，masks应该是黑白两色(0,1单通道，0代表背景，1代表前景。这里做2分类，0,1即标签)。
- 如果你需要处理其他数据格式，你可以在dataLoader.py中自定义DataSet

## train
```
python train.py --imgs_dir [imgs_path] --mask_dir [mask_path] --epochs [epochs] --load [pretrain weights] --scale [scale]
```

## predict
```
python predict.py --weights [model weights] --source [input data(file or dir)] --view_img [True or False] --save-mask [True or False] --project [output path]
```