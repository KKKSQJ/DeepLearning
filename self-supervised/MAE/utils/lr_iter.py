import math


def cosine_learning_rate(args, epoch, batch_iter, optimizer, train_batch):
    """Cosine Learning rate
    """
    total_epochs = args.max_epochs
    warm_epochs = args.warmup_epochs
    if epoch <= warm_epochs:
        lr_adj = (batch_iter + 1) / (warm_epochs * train_batch) + 1e-6
    else:
        lr_adj = 1/2 * (1 + math.cos(batch_iter * math.pi /
                                     ((total_epochs - warm_epochs) * train_batch)))

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * lr_adj
    return args.lr * lr_adj

def step_learning_rate(args, epoch, batch_iter, optimizer, train_batch):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    total_epochs = args.max_epochs
    warm_epochs = args.warmup_epochs
    if epoch <= warm_epochs:
        lr_adj = (batch_iter + 1) / (warm_epochs * train_batch)
    elif epoch < int(0.3 * total_epochs):
        lr_adj = 1.
    elif epoch < int(0.6 * total_epochs):
        lr_adj = 1e-1
    elif epoch < int(0.8 * total_epochs):
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr * lr_adj
    return args.lr * lr_adj