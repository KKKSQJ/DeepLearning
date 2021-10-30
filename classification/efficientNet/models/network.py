import torch.nn as nn
import torch
from torchsummary import summary
from torch import Tensor
from torch.nn import functional as F
from torchvision.models import AlexNet, resnet50

from typing import Optional, Callable
from collections import OrderedDict
from functools import partial
import math
import copy


# 确保所有层的通道数能被8整除
def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


# dropout层
class DropPath(nn.Sequential):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def drop_path(self, x, drop_prob: float = 0, training: bool = False):
        """
        Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
        "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

        This function is taken from the rwightman.
        It can be seen here:
        https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
        """
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)


# # 卷积+BN+激活函数
# class ConvBNAction(nn.Module):
#     def __init__(self,
#                  in_planes: int,
#                  out_planes: int,
#                  kernel_size: int = 3,
#                  stride: int = 1,
#                  groups: int = 1,
#                  norm_layer: Optional[Callable[..., nn.Module]] = None,
#                  activation_layer: Optional[Callable[..., nn.Module]] = None):
#         super(ConvBNAction, self).__init__()
#
#         padding = (kernel_size - 1) // 2
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if activation_layer is None:
#             activation_layer = nn.SiLU  # Swish (torch>=1.7)
#         self.conv = nn.Conv2d(in_channels=in_planes,
#                               out_channels=out_planes,
#                               kernel_size=kernel_size,
#                               stride=stride,
#                               padding=padding,
#                               groups=groups,
#                               bias=False)
#         self.bn = norm_layer(out_planes)
#         self.action = activation_layer()
#
#
#     def forward(self, x: Tensor) -> Tensor:
#         return self.action(self.bn(self.conv(x)))

# 卷积+BN+激活函数
class ConvBNAction(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):

        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU  # Swish (torch>=1.7)

        super(ConvBNAction, self).__init__(nn.Conv2d(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False),
            norm_layer(out_planes),
            activation_layer())


# SE模块
class SELayer(nn.Module):
    def __init__(self,
                 inp: int,
                 outp: int,
                 reduction: int = 4):
        super(SELayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(outp, _make_divisible(inp // reduction, 8), 1),
            nn.SiLU(),
            nn.Conv2d(_make_divisible(inp // reduction, 8), outp, 1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# MBCon模块的配置参数
class MBConvConfig:
    def __init__(self,
                 kernel: int,  # 3 or 5
                 input_c: int,
                 out_c: int,
                 expanded_ratio: int,  # 1 or 6 第一个1x1卷积的输出通道数，输入通道数的1倍或者6倍
                 stride: int,  # 1 or 2
                 use_se: bool,  # true
                 drop_rate: float,  # 在dropout层使用
                 index: str,  # 1a,2a,2b
                 width_coefficient: float  # 宽度调节因子
                 ):
        self.kernel = kernel
        self.input_c = self.adjust_channels(input_c, width_coefficient)
        self.expanded_c = self.input_c * expanded_ratio
        self.out_c = self.adjust_channels(out_c, width_coefficient)
        self.use_se = use_se
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index

    @staticmethod
    def adjust_channels(channles: int, width_coefficient: float):
        return _make_divisible(channles * width_coefficient, 8)


# MBConv模块
class MBConv(nn.Module):
    def __init__(self,
                 config: MBConvConfig,
                 norm_layer: Callable[..., nn.Module]):
        super(MBConv, self).__init__()
        assert config.stride in [1, 2], "illegal stride value."
        # 只有当MBCov结构的特征图与输出的特征图大小相同时候才使用shortcut
        self.use_res_connect = (config.stride == 1 and config.input_c == config.out_c)

        layers = OrderedDict()
        activation_layer = nn.SiLU

        # 第一个1x1卷积，他的卷积核个数是输入特征图通道数的n倍
        # n=1或者n=6，当n=1时，不要第一个1x1卷积
        # expand
        if config.expanded_c != config.input_c:
            layers.update({"expand_conv": ConvBNAction(
                config.input_c,
                config.expanded_c,
                kernel_size=1,
                stride=1,
                norm_layer=norm_layer,
                activation_layer=activation_layer
            )})

        # Depwise Conv
        layers.update({"dwconv": ConvBNAction(
            config.expanded_c,
            config.expanded_c,
            kernel_size=config.kernel,
            stride=config.stride,
            groups=config.expanded_c,  # group控制着每个卷积核与多少个输入通道进行运算，group=1则代表当前卷积核与全部输入通道进行运算。
            norm_layer=norm_layer,
            activation_layer=activation_layer
        )})

        # se模块
        if config.use_se:
            layers.update({"se": SELayer(config.input_c, config.expanded_c)})

        # project
        layers.update({"project_conv": ConvBNAction(
            config.expanded_c,
            config.out_c,
            kernel_size=1,
            stride=1,
            norm_layer=norm_layer,
            activation_layer=nn.Identity  # 这里是没有激活函数的
        )})

        self.block = nn.Sequential(layers)
        self.out_channels = config.out_c
        self.is_stride = config.stride > 1

        # 只有在使用shortcut连接时才使用dropout层
        if self.use_res_connect and config.drop_rate > 0:
            self.dropout = DropPath(config.drop_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        result = self.block(x)
        result = self.dropout(result)
        if self.use_res_connect:
            result += x

        return result


class EfficientNet(nn.Module):
    def __init__(self,
                 width_coefficient: float,
                 depth_coefficient: float,
                 num_classes: int = 1000,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None
                 ):
        super(EfficientNet, self).__init__()

        # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate, repeats
        # stage2 - stage8
        default_cnf = [[3, 32, 16, 1, 1, True, drop_connect_rate, 1],
                       [3, 16, 24, 6, 2, True, drop_connect_rate, 2],
                       [5, 24, 40, 6, 2, True, drop_connect_rate, 2],
                       [3, 40, 80, 6, 2, True, drop_connect_rate, 3],
                       [5, 80, 112, 6, 1, True, drop_connect_rate, 3],
                       [5, 112, 192, 6, 2, True, drop_connect_rate, 4],
                       [3, 192, 320, 6, 1, True, drop_connect_rate, 1]]

        def round_repeats(repeats):
            """Round number of repeats based on depth multiplier."""
            return int(math.ceil(repeats * depth_coefficient))

        if block is None:
            block = MBConv

        """
        partial:偏函数，可以扩展函数的功能。通道当我们频繁调用某个函数，且某些参数固定时使用
        # 类func = functools.partial(func, *args, **keywords) 
        func: 需要被扩展的函数，返回的函数其实是一个类 func 的函数
        *args: 需要被固定的位置参数
        **kwargs: 需要被固定的关键字参数
        # 如果在原来的函数 func 中关键字不存在，将会扩展，如果存在，则会覆盖
        """
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        adjust_channels = partial(MBConvConfig.adjust_channels, width_coefficient=width_coefficient)

        # 创建MBConv配置
        bneck_conf = partial(MBConvConfig, width_coefficient=width_coefficient)

        b = 0
        # 计算总共多少个MBConv层
        num_blocks = float(sum(round_repeats(i[-1]) for i in default_cnf))
        MBConv_setting = []
        for stage, args in enumerate(default_cnf):
            cnf = copy.copy(args)
            for i in range(round_repeats(cnf.pop(-1))):  # list.pop(index):返回list[index]. list剩下除index的元素
                if i > 0:
                    # 在堆叠MBConv模块的时候，除了每个stage的第一个模块之外的其他模型的步长都等于1，输入通道等于输出通道
                    # 即在每个stage开始的第一个MBConv模块进行下采样
                    cnf[-3] = 1  # stride
                    cnf[1] = cnf[2]  # 输入通道等于输出通道

                cnf[-1] = args[-2] * b / num_blocks
                index = str(stage + 1) + chr(i + 97)  # 1a,2a,2b,2c... 表示第几个stage的第几个MBConv模块
                MBConv_setting.append((bneck_conf(*cnf, index)))
                b += 1

        # create layers
        layers = OrderedDict()

        # first conv stage1
        # dict.update({key}:value)  ()里面是字典
        layers.update({"stem_conv": ConvBNAction(
            in_planes=3,
            out_planes=adjust_channels(32),
            kernel_size=3,
            stride=2,
            norm_layer=norm_layer
        )})

        # building MBConv block stage2 - stage 8
        for cnf in MBConv_setting:
            layers.update({cnf.index: block(cnf, norm_layer)})

        # build top stage 9
        last_conv_input_c = MBConv_setting[-1].out_c
        last_conv_output_c = adjust_channels(1280)
        layers.update({"top": ConvBNAction(
            in_planes=last_conv_input_c,
            out_planes=last_conv_output_c,
            kernel_size=1,
            norm_layer=norm_layer
        )})

        self.features = nn.Sequential(layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        classifier = []
        if dropout_rate > 0:
            classifier.append(nn.Dropout(p=dropout_rate, inplace=True))
        classifier.append(nn.Linear(last_conv_output_c, num_classes))
        self.classifier = nn.Sequential(*classifier)

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_imp(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_imp(x)


def efficientnet_b0(num_classes=1000):
    # input image size 224x224
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.0,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b1(num_classes=1000):
    # input image size 240x240
    return EfficientNet(width_coefficient=1.0,
                        depth_coefficient=1.1,
                        dropout_rate=0.2,
                        num_classes=num_classes)


def efficientnet_b2(num_classes=1000):
    # input image size 260x260
    return EfficientNet(width_coefficient=1.1,
                        depth_coefficient=1.2,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b3(num_classes=1000):
    # input image size 300x300
    return EfficientNet(width_coefficient=1.2,
                        depth_coefficient=1.4,
                        dropout_rate=0.3,
                        num_classes=num_classes)


def efficientnet_b4(num_classes=1000):
    # input image size 380x380
    return EfficientNet(width_coefficient=1.4,
                        depth_coefficient=1.8,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b5(num_classes=1000):
    # input image size 456x456
    return EfficientNet(width_coefficient=1.6,
                        depth_coefficient=2.2,
                        dropout_rate=0.4,
                        num_classes=num_classes)


def efficientnet_b6(num_classes=1000):
    # input image size 528x528
    return EfficientNet(width_coefficient=1.8,
                        depth_coefficient=2.6,
                        dropout_rate=0.5,
                        num_classes=num_classes)


def efficientnet_b7(num_classes=1000):
    # input image size 600x600
    return EfficientNet(width_coefficient=2.0,
                        depth_coefficient=3.1,
                        dropout_rate=0.5,
                        num_classes=num_classes)


if __name__ == '__main__':
    # test net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10
    model = efficientnet_b0(num_classes).to(device)
    img = torch.rand(1, 3, 224, 224).to(device)
    y = model(img)
    summary(model, input_size=(3, 224, 224))

    print(model)
    print(y)
