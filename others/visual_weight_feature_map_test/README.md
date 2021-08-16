# 本项目用于可视化分析
* 特征图
* 卷积核

## 目录结构
```
  ├── model: 网络结构目录
  ├── weights: 预训练模型目录
  ├── test_img: 测试图片目录
  visual_feature_map.py: 可视化特征图脚本
  visual_kernel_weight.py: 可视化卷积核脚本
```

## 预训练模型下载地址：
* resnet34:https://download.pytorch.org/models/resnet34-333f7ec4.pth
* alexnet:https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
* vgg16:https://download.pytorch.org/models/vgg16-397923af.pth

## 运行脚本
- 1.更改预训练模型路径以及transform类型
- 2.在网络结构中的forward函数中，输出想要可视化的特征层
- 3.python xx.py

## forward example
```
# resnet
# 将想要可视化的特征层加入output列表中

def _forward_impl(self, x):
    # See note [TorchScript super()]
    output = []
    x = self.conv1(x)
    output.append(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    output.append(x)
    x = self.layer2(x)
    output.append(x)
    x = self.layer3(x)
    output.append(x)
    x = self.layer4(x)
    output.append(x)
    #
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.fc(x)


    # return x
    return output

def forward(self, x):
    return self._forward_impl(x)
```

```
# alexnet
# 将可视化的特征层加入output列表中
def forward(self, x):
    # x = self.features(x)
    # x = self.avgpool(x)
    # x = torch.flatten(x, 1)
    # x = self.classifier(x)
    # return x
    output = []
    for name, module in self.features.named_children():
        x = module(x)
        if name in ["0", "3", "6"]:
            output.append(x)
    return output
```

## 关于pytorch中的~model.modules(), model.named_modules(), model.children(), model.named_children(), model.parameters(), model.named_parameters(), model.state_dict()
参考网址:https://www.jianshu.com/p/a4c745b6ea9b