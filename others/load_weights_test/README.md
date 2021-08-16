# 本项目用于测试 模型权重的 加载

## pytorch 权重加载API
```
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = xxx()
pre_train_weights = torch.load(model_weights_path, map_location=device)
net.load_state_dict(pre_train_weights, strict=False)
```

## 讲解PyTorch中~~model.modules(), model.named_modules(), model.children(), model.named_children(), model.parameters(), model.named_parameters(), model.state_dict()
参考网址：https://www.jianshu.com/p/a4c745b6ea9b
