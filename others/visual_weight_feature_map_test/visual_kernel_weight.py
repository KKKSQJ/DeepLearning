import torch
from model import resnet34,vgg16,alexnet
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def init_model(name:str='resnet',weights:str='./resnet34.pth'):
    assert os.path.exists(weights),"weights path: {} is not exists".format(weights)
    if name == 'resnet':
        model = resnet34(num_classes=1000)
        model.load_state_dict(torch.load(weights))
    elif name == 'alexnet':
        model = alexnet(num_classes=1000)
        model.load_state_dict(torch.load(weights))
    elif name == 'vgg':
        model = vgg16(num_classes=1000)
        model.load_state_dict(torch.load(weights))
    else:
        raise ValueError("model name must be 'resnet' or 'alexnet' or 'vgg'.now model name is {}".format(name))
    return model

def main(model):
    # model.state_dict(): 返回一个有序的字典
    # model.state_dict().keys(): 返回网络每一层的key
    # model.state_dict().values(): 返回对应key的参数值
    weights = model.state_dict()
    weights_keys = model.state_dict().keys()
    weights_values = model.state_dict().values()

    for key, value in model.state_dict().items():
        # remove num_batches_tracked para(in bn)
        if "num_batches_tracked" in key:
            continue
        # [kernel_number, kernel_channel, kernel_height, kernel_width]
        weight_t = model.state_dict()[key].numpy()

        # read a kernel information
        # k = weight_t[0, :, :, :]

        # calculate mean, std, min, max
        weight_mean = weight_t.mean()
        weight_std = weight_t.std(ddof=1)
        weight_min = weight_t.min()
        weight_max = weight_t.max()
        print("mean is {}, std is {}, min is {}, max is {}".format(weight_mean,
                                                                   weight_std,
                                                                   weight_max,
                                                                   weight_min))

        # plot hist image
        plt.close()
        weight_vec = np.reshape(weight_t, [-1])
        plt.hist(weight_vec, bins=50)
        plt.title(key)
        plt.show()

if __name__ == '__main__':
    model_name = 'resnet'
    weights = './weights/resnet34.pth'
    model = init_model(name=model_name,weights=weights)
    main(model=model)