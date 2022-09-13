import torch
import torch.nn as nn
from torch import Tensor

from typing import List, Callable


def shuffle_channels(x: Tensor, groups: int) -> Tensor:
    """shuffle channels of a 4-D Tensor"""
    batch_size, channels, height, width = x.size()

    assert channels % groups == 0

    channels_per_group = channels // groups

    # split into groups
    x = x.view(batch_size, groups, channels_per_group, height, width)

    # transpose 1, 2 axis
    x = x.transpose(1, 2).contiguous()

    # reshape into original
    x = x.view(batch_size, channels, height, width)
    return x


class ResidualBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int, groups: int):
        super(ResidualBlock, self).__init__()
        if stride not in [1, 2]:
            raise ValueError("illegal stride value")

        if stride == 1:
            assert inplanes == planes
        elif stride == 2:
            planes -= inplanes
            self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        assert planes % 4 == 0
        bottleneck_channels = planes // 4

        self.stride = stride
        self.groups = groups

        # 1x1GConv + BN + Relu
        self.group_conv1 = nn.Conv2d(inplanes, bottleneck_channels, kernel_size=1, stride=1, padding=0, groups=groups,
                                     bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.relu = nn.ReLU(inplace=True)

        # 3x3DWConv + BN
        self.depthwise_conv3 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride,
                                         padding=1, groups=bottleneck_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)

        # 1x1GConv + BN
        self.group_conv = nn.Conv2d(bottleneck_channels, planes, kernel_size=1, stride=1, padding=0, groups=groups,
                                    bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

    def forward(self, x: Tensor) -> Tensor:
        out = self.group_conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = shuffle_channels(out, groups=self.groups)

        out = self.depthwise_conv3(out)
        out = self.bn2(out)

        out = self.group_conv(out)
        out = self.bn3(out)

        if self.stride == 2:
            x = self.avg_pool(x)
            out = torch.cat([x, out], dim=1)

        elif self.stride == 1:
            out = x + out

        out = self.relu(out)
        return out


class ShuffleNetv1(nn.Module):
    def __init__(self,
                 stages_repeats: List[int] = [3, 7, 3],
                 stages_out_channels: List[int] = [3, 24, 240, 480, 960],
                 groups: int = 3,
                 ratio: float = 1.,
                 num_classes: int = 1000):

        super(ShuffleNetv1, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError("expected stages_repeats as list of 3 positive ints")

        if len(stages_out_channels) != 5:
            raise ValueError("expected stages_out_channels as list of 5 positive ints")

        stages_out_channels = [int(x * ratio) for x in stages_out_channels]

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, stages_out_channels[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stages_out_channels[1]),
            nn.ReLU(inplace=True)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # NOTE: Do not use group convolution for the first conv1x1 in Stage 2.
        self.stage2 = self._make_stage(ResidualBlock, stages_out_channels[1], stages_out_channels[2],
                                       blocks=stages_repeats[0], stride=2, groups=groups, conv_group=False)
        self.stage3 = self._make_stage(ResidualBlock, stages_out_channels[2], stages_out_channels[3],
                                       blocks=stages_repeats[1], stride=2, groups=groups)
        self.stage4 = self._make_stage(ResidualBlock, stages_out_channels[3], stages_out_channels[4],
                                       blocks=stages_repeats[2], stride=2, groups=groups)

        self.fc = nn.Linear(stages_out_channels[4], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = x.mean([2, 3])  # gloabl pool
        x = self.fc(x)

        return x

    # NOTE: Do not use group convolution for the first conv1x1 in Stage 2.
    def _make_stage(self, block, inplanes: int, planes: int, blocks: int, stride: int = 2,
                    groups: int = 3, conv_group=True) -> nn.Sequential:

        layers = [block(inplanes, planes, stride=stride, groups=groups if conv_group else 1)]

        for _ in range(blocks):
            layers.append(block(planes, planes, stride=1, groups=groups))

        return nn.Sequential(*layers)


def shufflenet_v1_x1_g1(num_classes=1000, ratio: float = 1.0):
    model = ShuffleNetv1(stages_repeats=[3, 7, 3],
                         stages_out_channels=[3, 24, 144, 288, 576],
                         groups=1, ratio=ratio, num_classes=num_classes)

    return model


def shufflenet_v1_x1_g2(num_classes=1000, ratio: float = 1.0):
    model = ShuffleNetv1(stages_repeats=[3, 7, 3],
                         stages_out_channels=[3, 24, 200, 400, 800],
                         groups=2, ratio=ratio, num_classes=num_classes)

    return model


def shufflenet_v1_x1_g3(num_classes=1000, ratio: float = 1.0):
    model = ShuffleNetv1(stages_repeats=[3, 7, 3],
                         stages_out_channels=[3, 24, 240, 480, 960],
                         groups=3, ratio=ratio, num_classes=num_classes)

    return model


def shufflenet_v1_x1_g4(num_classes=1000, ratio: float = 1.0):
    model = ShuffleNetv1(stages_repeats=[3, 7, 3],
                         stages_out_channels=[3, 24, 272, 544, 1088],
                         groups=4, ratio=ratio, num_classes=num_classes)

    return model


def shufflenet_v1_x1_g8(num_classes=1000, ratio: float = 1.0):
    model = ShuffleNetv1(stages_repeats=[3, 7, 3],
                         stages_out_channels=[3, 24, 384, 768, 1536],
                         groups=8, ratio=ratio, num_classes=num_classes)

    return model


def get_model(name):
    return model_dict[name]


model_dict = {
    "shufflenet_v1_g1": shufflenet_v1_x1_g1,
    "shufflenet_v1_g2": shufflenet_v1_x1_g2,
    "shufflenet_v1_g3": shufflenet_v1_x1_g3,
    "shufflenet_v1_g4": shufflenet_v1_x1_g4,
    "shufflenet_v1_g8": shufflenet_v1_x1_g8,
}

if __name__ == '__main__':
    model = shufflenet_v1_x1_g3(num_classes=5, ratio=1)
    x = torch.rand(1, 3, 224, 224)
    y = model(x)
    print(model)
    print(y)

    from thop import profile
    from thop import clever_format

    flops, params = profile(model, inputs=(x,))

    flops, params = clever_format([flops, params], "%.3f")
    print(f"FLOPs:{flops}")
    print(f"Params:{params}")
