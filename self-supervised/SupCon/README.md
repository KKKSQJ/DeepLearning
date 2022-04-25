# SupCon
论文：[Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362)
- 注：windows下运行失败，因为faiss对windows的支持不够友善

## Training
In order to execute Cifar10 training run:
```
python train.py --config_name configs/train/train_supcon_resnet18_cifar10_stage1.yml
python swa.py --config_name configs/train/swa_supcon_resnet18_cifar10_stage1.yml
python train.py --config_name configs/train/train_supcon_resnet18_cifar10_stage2.yml
python swa.py --config_name configs/train/swa_supcon_resnet18_cifar10_stage2.yml
```

In order to run LRFinder on the second stage of the training, run:
```
python learning_rate_finder.py --config_name configs/train/lr_finder_supcon_resnet18_cifar10_stage2.yml
```

The process of training Cifar100 is exactly the same, just change config names from cifar10 to cifar100. 

After that you can check the results of the training either in `logs` or `runs` directory. For example, in order to check tensorboard logs for the first stage of Cifar10 training, run:
```
tensorboard --logdir runs/supcon_first_stage_cifar10
```

## Visualizations 

This repo is supplied with t-SNE visualizations so that you can check embeddings you get after the training. Check `t-SNE.ipynb` or `t-SNE.py` for details. 