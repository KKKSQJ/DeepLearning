# EfficientNet

## 目录结构
```
efficientNet:.
│  class_indices.json   # 存放类别数据，索引：类别
│  README.md            
│  test.py              # 推理脚本
│  train.py             # 训练脚本
│  trans_weights_to_pytorch.py      # 获取预训练权重脚本，将tf模型转为pytorch模型
│  utils.py             # 其他
│  
├─data                  # 存放数据
│  ├─train              # 存放训练数据
│  │  ├─classes1_dir    # 存放类别1的图片
│  │  ├─classes2_dir
│  ├─val                # 存放测试数据
│  │  ├─classes1656_dir
│  │  ├─classes28565_dir
│      data.txt
│      
├─dataLoader
│  │  dataLoader.py     
│  │  dataSet.py
│  │  
│          
├─doc
│      efficientB0-B7.png
│      efficientB0.png
│      MBConv.png
│      SE.png
│      
├─models
│  │  network.py    # 网络结构
│  │  
│          
├─pre_train_model    # 存放预训练模型，xx.pth由trans_weights_to_pytorch.py生成
│      efficientnetb0.pth
│      pre_train_model.txt
│      
├─runs               # 训练生成的日志文件
│  ├─Oct29_18-39-49_LAPTOP-FV221P37
│  │  │  classes.jpg
│  │  │  class_indices.json
│  │  │  events.out.tfevents.1635503991.LAPTOP-FV221P37.1820.0
│  │  │  train.txt
│  │  │  val.txt
│  │  │  
│  │  └─weights     # 训练生成的模型
│  │          best_model.pth
│  │          model_0.pth
│  │          
│  └─result         # 测试结果
│          result.txt
```

## 数据集链接
花分类数据集

链接：https://pan.baidu.com/s/164u93bji6A4W_r6fZkKhwg 
提取码：uage

## 训练
```
python train.py --data-path [data_path] --weights [weights_path]
```

## 推理
```
python test.py --weights [weights_path] --source [data_path] --view-img True
```

## tensorboard
```
tensorboard --logdir path-to-runs
```