from torch import optim as optim
from torch.optim import SGD, Adam, AdamW


class Optimizer(object):
    def __init__(self, name) -> None:
        super().__init__()
        self._name = name

    def __call__(self, param, lr, weight_decay, finetune=False):
        if self._name.lower() == "sgd":
            optimizer = SGD(
                param,
                lr,
                weight_decay=weight_decay,
                momentum=0.9,
            )
        elif self._name.lower() == "adam":
            optimizer = Adam(
                param,
                lr,
                weight_decay=weight_decay
            )
        elif self._name.lower() == "adamw":
            if finetune:
                optimizer = AdamW(
                    param,
                    lr,
                    betas=(0.9, 0.999),
                    weight_decay=weight_decay
                )
            else:
                optimizer = AdamW(
                    param,
                    lr,
                    betas=(0.9, 0.95),
                    weight_decay=weight_decay
                )
        else:
            raise NotImplementedError(f"{self._name} optimizer have not been implement!")

        return optimizer


def build_optimizer(model, opt_name, lr, weights_decay):
    """Build optimizer, set weight decay of normalization to 0 by default
    """
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    optimizer = None
    if opt_name.lower() == 'sgd':
        optimizer = optim.SGD(parameters,
                              momentum=0.9,
                              nesterov=False,
                              lr=lr,
                              weight_decay=weights_decay)
    elif opt_name.lower() == 'adamw':
        optimizer = optim.AdamW(parameters,
                                lr=lr,
                                weight_decay=weights_decay)

    return optimizer


def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin