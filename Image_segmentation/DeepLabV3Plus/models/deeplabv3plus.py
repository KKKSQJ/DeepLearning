import os
import os.path
from typing import Dict, List
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.resnet_backbone import resnet50, resnet101
from models.mobilenet_backbone import mobilenet_v3_large


class DeepLabv3(nn.Module):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.
    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    __constants__ = ['aux_classifier']

    def __init__(self, backbone, classifier, aux_classifier=None):
        super(DeepLabv3, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # x.shape:B C H W
        input_shape = x.shape[-2:]  # (H,W)
        features = self.backbone(x)

        result = OrderedDict()
        # x = features["out"]
        x = self.classifier(features)

        # 使用双线性插值还原回原图尺度
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            # 使用双线性插值还原回原图尺度
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result


class IntermediateLayerGetter(nn.ModuleDict):
    """
    模块封装器，从一个模型中返回中间层

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Args:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    _version = 2
    __annotations__ = {
        "return_layer": Dict[str, str]
    }

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layer = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没有使用到的模块全部删掉
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layer:
                del return_layer[name]
            if not return_layer:
                break

        super(IntermediateLayerGetter, self).__init__(layers)  # 将layers注册到类中
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():  # 因为已经将layers（模型）注册到类中了，所以可以调用self.items()
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        inter_channels = in_channels // 4
        super(FCNHead, self).__init__(
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Conv2d(inter_channels, out_channels, 1)
        )


class DeepLabHeadv3Plus(nn.Module):
    def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(DeepLabHeadv3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.aspp = ASPP(in_channels, aspp_dilate, 256)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),  # 304=48+256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project(feature["low_level"])
        output_feature = self.aspp(feature["out"])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rate: List[int], out_channels: int = 256):
        super(ASPP, self).__init__()

        modules = [nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )]

        rates = tuple(atrous_rate)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

    def forward(self, x: Tensor) -> Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, rate):
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=rate, dilation=rate, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x: Tensor) -> Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


def deeplabv3plus_resnet101(aux, num_classes=21, output_stride=8, pretrain_backbone=False, pretrain_weights=None):
    # pretrain weights
    # https://share.weiyun.com/UNPZr3dk
    if output_stride == 8:
        backbone = resnet101(replace_stride_with_dilation=[False, True, True])
        aspp_dilate = [12, 24, 36]
    else:
        backbone = resnet101(replace_stride_with_dilation=[False, False, True])
        aspp_dilate = [6, 12, 18]

    if pretrain_backbone:
        assert os.path.exists(pretrain_weights), f"{pretrain_weights} dose not exists!"
        backbone.load_state_dict(torch.load(pretrain_weights, map_location="cpu"))

    out_inplanes = 2048
    aux_inplanes = 1024
    low_level_planes = 256

    return_layers = {"layer4": "out", "layer1": "low_level"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = DeepLabHeadv3Plus(out_inplanes, low_level_planes, num_classes, aspp_dilate)

    model = DeepLabv3(backbone, classifier, aux_classifier)
    return model


def deeplabv3plus_resnet50(aux, num_classes=21, output_stride=8, pretrain_backbone=False, pretrain_weights=None):
    # pretrain weights
    # https://share.weiyun.com/uTM4i2jG
    if output_stride == 8:
        backbone = resnet50(replace_stride_with_dilation=[False, True, True])
        aspp_dilate = [12, 24, 36]
    else:
        backbone = resnet50(replace_stride_with_dilation=[False, False, True])
        aspp_dilate = [6, 12, 18]

    if pretrain_backbone:
        assert os.path.exists(pretrain_weights), f"{pretrain_weights} dose not exists!"
        backbone.load_state_dict(torch.load(pretrain_weights, map_location="cpu"))

    out_inplanes = 2048
    aux_inplanes = 1024
    low_level_planes = 256

    return_layers = {"layer4": "out", "layer1": "low_level"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = DeepLabHeadv3Plus(out_inplanes, low_level_planes, num_classes, aspp_dilate)

    model = DeepLabv3(backbone, classifier, aux_classifier)
    return model


def deeplabv3plus_mobilenetv3_large(aux, num_classes=21, output_stride=8, pretrain_backbone=False,
                                    pretrain_weights=None):
    # pretrain
    # https://share.weiyun.com/djX6MDwM
    backbone = mobilenet_v3_large(dilated=True)

    if pretrain_backbone:
        # 载入mobilenetv3 large backbone预训练权重
        assert os.path.exists(pretrain_weights), f"{pretrain_weights} dose not exists!"
        backbone.load_state_dict(torch.load(pretrain_weights, map_location='cpu'))

    backbone = backbone.features

    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "is_strided", False)] + [len(backbone) - 1]
    out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
    out_inplanes = backbone[out_pos].out_channels
    aux_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
    aux_inplanes = backbone[aux_pos].out_channels
    low_level_pos = stage_indices[-5]
    low_level_inplanes = backbone[low_level_pos].out_channels
    return_layers = {str(out_pos): "out", str(low_level_pos): "low_level"}
    if aux:
        return_layers[str(aux_pos)] = "aux"

    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    # why using aux: https://github.com/pytorch/vision/issues/4292
    if aux:
        aux_classifier = FCNHead(aux_inplanes, num_classes)

    classifier = DeepLabHeadv3Plus(out_inplanes, low_level_inplanes, num_classes, aspp_dilate=[12, 24, 36])

    model = DeepLabv3(backbone, classifier, aux_classifier)

    return model


if __name__ == '__main__':
    x = torch.randn(2, 3, 480, 480)
    model = deeplabv3plus_resnet50(aux=True, num_classes=21, output_stride=8, pretrain_backbone=False)
    y = model(x)
    print(y["out"].shape)
