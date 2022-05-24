# 论文《TransFG: A Transformer Architecture for Fine-Grained Recognition》 复现

论文讲解：https://zhuanlan.zhihu.com/p/519173447
## 数据集
- [CUB-200-2011](http://www.vision.caltech.edu/datasets/cub_200_2011/)
- [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
- [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/)
- [NABirds](https://dl.allaboutbirds.org/nabirds)
- [iNaturalist 2017](https://github.com/visipedia/inat_comp/tree/master/2017)

## 预训练模型
[模型链接](https://console.cloud.google.com/storage/browser/vit_models) ViT-B_16, ViT-B_32...
```
wget https://storage.googleapis.com/vit_models/imagenet21k/{MODEL_NAME}.npz
```

注意：在本项目中，我只验证了CUB-200-2011数据集。如果你想要训练你自己的数据，你可以在dataLoader/dataset.py中diy你的数据，然后把它在utils/builder.py下的build_loader函数中进行加载。

## Train
- 注：下载好数据集之后，在配置文件中（config/example.yaml）配置你的训练信息，详情见配置文件的注解。

```
python train.py --config_name config/example.yaml
```

## Test
- 注： 检查你是否生成class_indices.json。该json包含标签和类别名字的映射关系。
- For Example:
```
{
    "0": "001.Black_footed_Albatross",
    "1": "002.Laysan_Albatross",
}
```

```
python test.py --cfg xxx.yaml --weights xxx.pth --source test_data_path --class-indices xxx.json
```