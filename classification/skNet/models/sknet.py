import torch
from torch import nn
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional


class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, M=2, G=32, r=2, stride=1, L=32):
        """ Constructor
        Args:
            in_channels: input channel dimensionality.
            out_channels: output channel dimensionality.
            M: the number of branches.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(in_channels // r), L)
        self.M = M
        self.out_channels = out_channels
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(out_channels, d)

        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, out_channels)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v.contiguous()


# official
class SKUnit(nn.Module):
    def __init__(self, in_features, out_features, M, G, r, mid_features=None, stride=1, L=32):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            mid_features: the channel dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()
        if mid_features is None:
            mid_features = int(out_features / 2)
        self.feas = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1),
            nn.BatchNorm2d(mid_features),
            SKConv(mid_features, mid_features, M, G, r, stride=stride, L=L),
            nn.BatchNorm2d(mid_features),
            nn.Conv2d(mid_features, out_features, 1, stride=1),
            nn.BatchNorm2d(out_features)
        )
        if in_features == out_features:  # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential()
        else:  # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride),
                nn.BatchNorm2d(out_features)
            )

    def forward(self, x):
        fea = self.feas(x)
        return fea + self.shortcut(x)


class SKBlock(nn.Module):
    '''
    基于Res Block构造的SK Block
    ResNeXt有  1x1Conv（通道数：x） +  SKConv（通道数：x）  + 1x1Conv（通道数：2x） 构成
    '''

    """ Constructor
    Args:
        inplanes: input channel dimensionality.
        planes: output channel dimensionality.
        M: the number of branchs.
        G: num of convolution groups.
        r: the radio for compute d, the length of z.
        stride: stride.
        L: the minimum dim of the vector z in paper.
    """

    expansion = 2

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            M: int = 2,
            G: int = 32,
            r: int = 16,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(SKBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, 1, 0, bias=False),
            norm_layer(planes),
            nn.ReLU(inplace=True)
        )

        self.conv2 = SKConv(planes, planes, M, G, r, stride)

        self.conv3 = nn.Sequential(
            nn.Conv2d(planes, planes * self.expansion, 1, 1, 0, bias=False),
            norm_layer(planes * self.expansion)
        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out += shortcut
        out = self.relu(out)
        return out


class SKNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[SKBlock]] = SKBlock,
            layers: List[int] = [3, 4, 6, 3],
            num_classes: int = 1000,
            M: int = 2,
            G: int = 32,
            r: int = 16,
            stem_width: int = 64,
            deep_stem: bool = False,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(SKNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = stem_width * 2 if deep_stem else 64

        if deep_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(stem_width),
                nn.ReLU(inplace=True),
                nn.Conv2d(stem_width, stem_width * 2, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 128, layers[0], stride=1, M=M, G=G, r=r)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2, M=M, G=G, r=r)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2, M=M, G=G, r=r)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2, M=M, G=G, r=r)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # GAP全局池化
        self.fc = nn.Linear(1024 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def _make_layer(self, block: Type[Union[SKBlock]], planes: int, blocks: int, stride: int = 1, M: int = 2,
                    G: int = 32, r: int = 16) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, 0, bias=False),
                norm_layer(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, M, G, r, norm_layer))

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, M=M, G=G, r=r, norm_layer=norm_layer))

        return nn.Sequential(*layers)


if __name__ == '__main__':
    # x = torch.rand(8, 64, 32, 32)
    # conv = SKConv(64, 64, 3, 8, 2)
    # out = conv(x)
    # criterion = nn.L1Loss()
    # loss = criterion(out, x)
    # loss.backward()
    # print('out shape : {}'.format(out.shape))
    # print('loss value : {}'.format(loss))

    x = torch.rand(1, 3, 224, 224)
    model = SKNet()
    y = model(x)
    print(y.shape)
