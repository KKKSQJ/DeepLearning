# BDB
该代码复现论文《Batch DropBlock Network for Person Re-identification and Beyond》

## 代码说明
该代码用于图像检索

## 数据集说明
数据集格式参考
```
--data(存放数据的目录)
----|
--------query(查询图)
--------gallery(检索库)
--------train(训练数据)

其中，train里面存放的不同类的文件夹，每个文件夹下面存放对应类的图片
类别文件夹名字：id_name
For example: 0_people
             1_dog
```

## 训练
```
python train.py --data-path='./data' --batch-size=256 
```

## 推理
```
python inference.py
```