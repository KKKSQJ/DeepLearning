import os
import random
import numpy as np

import torch
import torch.optim as optim
import torch_optimizer as jettify_optim

from losses.loss import LOSSES


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def add_to_logs(logging, message):
    logging.info(message)


def add_to_tensorboard_logs(writer, message, tag, index):
    writer.add_scalar(tag, message, index)


def copy_parameters_from_model(model):
    copy_of_model_parameters = [p.clone().detach() for p in model.parameters() if p.requires_grad]
    return copy_of_model_parameters


def copy_parameters_to_model(copy_of_model_parameters, model):
    for s_param, param in zip(copy_of_model_parameters, model.parameters()):
        if param.requires_grad:
            param.data.copy_(s_param.data)


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
}


def build_optim(model, optimizer_params, scheduler_params, loss_params):
    if 'params' in loss_params:
        criterion = LOSSES[loss_params['name']](**loss_params['params'])
    else:
        criterion = LOSSES[loss_params['name']]()

    optimizer = OPTIMIZERS[optimizer_params["name"]](model.parameters(), **optimizer_params["params"])

    if scheduler_params:
        scheduler = SCHEDULERS[scheduler_params["name"]](optimizer, **scheduler_params["params"])
    else:
        scheduler = None

    return {"criterion": criterion, "optimizer": optimizer, "scheduler": scheduler}
