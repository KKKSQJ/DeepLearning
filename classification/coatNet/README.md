# coatNet

## 数据集下载
花分类数据集

链接：https://pan.baidu.com/s/164u93bji6A4W_r6fZkKhwg 
提取码：uage

## train
```
python train.py --data-path [data_path] 
```

## test
```
python test.py --weights [weights_path] --source [data_path] --class-indices [json_path] --view-img True
```

## tensorboard
```
tensorboard --logdir path-to-runs
```