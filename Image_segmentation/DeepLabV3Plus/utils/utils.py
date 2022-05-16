import random
import os
import numpy as np

import torch
import torch.optim as optim
import torch_optimizer as jettify_optim


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


OPTIMIZERS = {
    "Adam": optim.Adam,
    'AdamW': optim.AdamW,
    "SGD": optim.SGD,
    'LookAhead': jettify_optim.Lookahead,
    'Ranger': jettify_optim.Ranger,
    'RAdam': jettify_optim.RAdam,
}

SCHEDULERS = {
    "ReduceLROnPlateau": optim.lr_scheduler.ReduceLROnPlateau,
    "CosineAnnealingLR": optim.lr_scheduler.CosineAnnealingLR,
    "LambdaLR": optim.lr_scheduler.LambdaLR
}

LOSSES = {
    'CrossEntropy': torch.nn.CrossEntropyLoss,
}


def build_optim(params_to_optimize, optimizer_params, loss_params, scheduler_params):
    if 'params' in loss_params:
        weight = loss_params['params']['weight']
        if weight is not None:
            loss_params["params"]["weight"] = torch.FloatTensor(loss_params['params']['weight']).cuda()
        criterion = LOSSES[loss_params['name']](**loss_params['params'])
    else:
        criterion = LOSSES[loss_params['name']]()

    optimizer = OPTIMIZERS[optimizer_params["name"]](params_to_optimize, **optimizer_params["params"])

    if scheduler_params:
        scheduler = SCHEDULERS[scheduler_params["name"]](optimizer, **scheduler_params["params"])
    else:
        scheduler = None

    return {"criterion": criterion, "optimizer": optimizer, "scheduler": scheduler}


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def lr_scheduler(
        num_step: int,
        epochs: int,
        warmup=True,
        warmup_epochs=1,
        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return f


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg
