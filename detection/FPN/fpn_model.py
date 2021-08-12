'''FPN in PyTorch.
See the paper "Feature Pyramid Networks for Object Detection" for more details.
'''

import torch
import torch.nn as nn
from torch.jit.annotations import List,Dict
from torchvision.ops.misc import FrozenBatchNorm2d
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

"""
Resnet18,resnet32的组件
由
conv(3x3,s=stride,p=1)+bn+relu  注意：当stride=2，进行下采样，一般在每个stage的第一个卷积层中s=2
conv(1x1,s=1,p=0)+bn
组成
"""
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1,stride=stride,bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


"""
Resnet50及以上的组件
由
conv(3x3,s=stride,p=1)+bn+relu  注意：当stride=2，进行下采样，一般在每个stage(虚线)的第一个卷积层中s=2
conv(1x1,s=1,p=0)+bn+relu
conv(1x1,s=1,p=0)+bn
组成
"""
class Bottleneck(nn.Module):
    expansion=4
    def __init__(self, inplanes, planes, stride=1, downsample=None,norm_layer=None,base_width=64):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=1,stride=1,bias=False) # 因为加了BatchNorm，所以bias可以设置为False
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, width * self.expansion,kernel_size=1,stride=1,bias=False)
        self.bn3 = norm_layer(width * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# backbone为resnet50的FPN网络
class FPN(nn.Module):
    def __init__(self,
                 block=Bottleneck,      # 残差块
                 layers= [3,4,6,3],     # 每个stage的残差块个数
                 fpn_channel = 265,     # fpn模块输出通道
                 norm_layer=None, #FrozenBatchNorm2d,  # 如果norm_layer=None，则norm_layer=nn.BatchNorm2d
                 zero_init_residual=False,
                 extra_block = True,            # true:fpn增加额外的feature map[p6,p7]
                 include_top = False,           # true:单纯resnet分类网络
                 num_classes = 1000):           # 类别数量
        super(FPN, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self._extra_block = extra_block
        self._include_top = include_top

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3,self.in_planes,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = norm_layer(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        # Bottom-top layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)


        # Smooth layers
        self.smooth1 = nn.Conv2d(fpn_channel, fpn_channel, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(fpn_channel, fpn_channel, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(fpn_channel, fpn_channel, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(512 * block.expansion, fpn_channel, kernel_size=1, stride=1)
        self.latlayer2 = nn.Conv2d(256 * block.expansion, fpn_channel, kernel_size=1, stride=1)
        self.latlayer3 = nn.Conv2d(128 * block.expansion, fpn_channel, kernel_size=1, stride=1)

        if extra_block:
            self.p6 = nn.Conv2d(fpn_channel, fpn_channel, kernel_size=3, stride=2, padding=1)
            self.p7 = nn.Conv2d(fpn_channel, fpn_channel, kernel_size=3, stride=2, padding=1)
            for module in [self.p6, self.p7]:
                nn.init.kaiming_uniform_(module.weight, a=1)
                nn.init.constant_(module.bias, 0)

        if include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)




    def _make_layer(self,block,planes,blocks,stride=1):
        norm_layer = self._norm_layer
        downsample = None

        # 如果stride=2，即进行下采样（2,3,4,stage的第一个卷积层进行2倍下采样从操作，图上虚线连接）
        # 如果self.in_planes != planes * block.expansion,即第一个stage残差连接
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion)
            )

        layers = []
        # 如果stride=2，即进行下采样（2,3,4,stage的第一个卷积层进行2倍下采样从操作，图上虚线连接）
        # 如果self.in_planes != planes * block.expansion,即第一个stage残差连接
        layers.append(block(self.in_planes, planes, stride, downsample, norm_layer))
        self.in_planes = planes * block.expansion

        # 添加每个stage里中除第一个block之外的其他block。
        # 如resnet50,每个stage的block数量为[3,4,6,3]。
        # 除却每个stage的第一个block需要downsample(既改变通道数，又改变size),其他block只需要改变通道数，不改变图片size
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        B, C, H, W = y.size()
        # (2x)上采样，通过线性插值实现
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        c1 = self.relu(self.bn1(self.conv1(x)))
        c1 = self.maxpool(c1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        if self._include_top:
            out = self.avgpool(c5)
            out = torch.flatten(out, 1)
            out = self.fc(out)
            return out

        # Top-down
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p3 = self._upsample_add(p4, self.latlayer3(c3))

        # Smooth
        p5 = self.smooth1(p5)
        p4 = self.smooth2(p4)
        p3 = self.smooth1(p3)

        if self._extra_block:
            p6 = self.p6(p5)
            p7 = self.p7(self.relu(p6))
            return p7, p6, p5, p4, p3

        return p5, p4, p3


def  resnet50_backbone_with_fpn():
    return FPN(Bottleneck, [3,4,6,3], extra_block=True, include_top=False)

if __name__ == '__main__':
    # resnet50_backbone_with_fpn = FPN(Bottleneck, [3,4,6,3])
    net = resnet50_backbone_with_fpn()
    print(net)

    fms = net(Variable(torch.randn(1,3,640,640)))
    for index, fm in enumerate(fms):
        print("feature map :p{} shape is {}".format(7-index, fm.size()))


    import torchvision
    from torchsummary import summary
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    summary(net, input_size=(3, 640, 640))






