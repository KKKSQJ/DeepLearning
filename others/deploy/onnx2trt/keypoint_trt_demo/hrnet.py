import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


# 输入：从上一层接收分支（n branch）
# 处理过程：对应分支进行处理（identity或上采样或下采样）
# 输出：输出每个分支(n branch)
class StageModule(nn.Module):
    """
    构建对应stage，即用来融合不同尺度的实现
    :param input_branches: 输入的分支数，每个分支对应一种尺度
    :param output_branches: 输出的分支数
    :param c: 输入的第一个分支通道数,后面的分支的通道数为第一个分支的c*2^(n-1)
    """

    def __init__(self, input_branches: int, out_branches: int, c: int):
        super(StageModule, self).__init__()
        self.input_branches = input_branches
        self.out_branches = out_branches
        self.c = c

        # 构造每个分支的模块
        self.branches = nn.ModuleList()
        for i in range(self.input_branches):  # 每个分支都先通过4个BasicBlock
            in_c = c * (2 ** i)
            branch = nn.Sequential(
                BasicBlock(in_c, in_c),
                BasicBlock(in_c, in_c),
                BasicBlock(in_c, in_c),
                BasicBlock(in_c, in_c)
            )
            self.branches.append(branch)

        # 融合每个分支的输出
        self.fuse_layers = nn.ModuleList()
        # 下面的代码可以理解为第j个分支 经过一系列操作 输出 用于 融合第i的分支的模块
        for i in range(self.out_branches):  # 构建每个分支的操作：如identity，上采样，下采样
            self.fuse_layers.append(nn.ModuleList())  # 用于存放 融合第i个分支的所需的layer
            for j in range(self.input_branches):  # 遍历每个输入分支，构建该分支到输出的操作，如identity，上采样，下采样
                if j == i:  # 相等，说明是同一分支，不做任何操作
                    self.fuse_layers[-1].append(nn.Identity())
                elif j < i:  # 输入分支分辨率大于输出分支分辨率，需要下采样
                    # 每次只能下采样2倍，通过一个3x3卷积实现。下采样4倍就是2个3x3卷积
                    # 这里的下采样操作分为两个部分，前i-j-j个卷积层，只进行下采样，通道不变。最后一个卷积层，既要下采样，又要变换通道
                    ops = []
                    for k in range(i - j - 1):  # 前i-j-j个卷积层
                        ops.append(nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** j)),
                            nn.ReLU(inplace=True)
                        ))
                    # 最后一个卷积层，既要进行下采样，又要进行通道变化
                    ops.append(nn.Sequential(
                        # 将通道数量变为输出分支的数量
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(c * (2 ** i)),
                        nn.ReLU(inplace=True)
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))
                else:  # j>i 输入分支的分辨率小于输出分支的分辨率，要进行上采样
                    self.fuse_layers[-1].append(nn.Sequential(
                        # 上采样操作通过1x1卷积改变通道，通过upsample上采样
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, padding=0, bias=False),
                        nn.BatchNorm2d(c * (2 ** i)),
                        nn.Upsample(scale_factor=2.0 ** (j - i), mode='nearest')
                    ))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 每个分支都通过对应的block
        x = [branch(xi) for branch, xi in zip(self.branches, x)]

        # 融合不同尺度的特征图
        x_fused = []
        for i in range(len(self.fuse_layers)):
            x_fused.append(
                self.relu(
                    # 融合第i个输出分支  所需的j个输入
                    sum([self.fuse_layers[i][j](x[j]) for j in range(len(self.branches))])
                )
            )
        return x_fused


class HighResolution(nn.Module):
    def __init__(self, base_channel: int = 32, num_joint: int = 17, stage_block=None):
        super(HighResolution, self).__init__()

        if stage_block is None:
            stage_block = [1, 4, 2]

        # stem  下采样4倍
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Stage1
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(256)
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, stride=1, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
        )

        # transition1
        self.transition1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(256, base_channel, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel),
                    nn.ReLU(inplace=True)
                ),
                nn.Sequential(
                    nn.Sequential(
                        nn.Conv2d(256, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(base_channel * 2),
                        nn.ReLU(inplace=True)
                    )
                )
            ]
        )

        # Stage2
        self.stage2 = nn.Sequential(
            *[StageModule(input_branches=2, out_branches=2, c=base_channel) for _ in range(stage_block[0])])

        # transition2
        self.transition2 = nn.Sequential(
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 4),
                    nn.ReLU(inplace=True)
                )
            )
        )

        # stage3
        self.stage3 = nn.Sequential(
            *[StageModule(input_branches=3, out_branches=3, c=base_channel) for _ in range(stage_block[1])])

        # transition3
        self.transition3 = nn.ModuleList(
            [
                nn.Identity(),
                nn.Identity(),
                nn.Identity(),
                nn.Sequential(
                    nn.Sequential(
                        nn.Conv2d(base_channel * 4, base_channel * 8, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(base_channel * 8),
                        nn.ReLU(inplace=True)
                    )
                )
            ]
        )

        # Stage4
        self.stage4 = nn.Sequential(
            StageModule(input_branches=4, out_branches=4, c=base_channel),
            StageModule(input_branches=4, out_branches=4, c=base_channel),
            StageModule(input_branches=4, out_branches=1, c=base_channel)

        )

        # 预测头
        self.final_layer = nn.Conv2d(base_channel, num_joint, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]

        x = self.stage2(x)
        x = [self.transition2[i](x[i]) for i in range(len(x))]
        x.append(self.transition2[-1](x[-1]))  # 下一分支的下采样

        # x = [
        #     self.transition2[0](x[0]),
        #     self.transition2[1](x[1]),
        #     self.transition2[2](x[-1])
        # ]  # New branch derives from the "upper" branch only

        x = self.stage3(x)
        x = [self.transition3[i](x[i]) for i in range(len(x))]
        x.append(self.transition3[-1](x[-1]))  # 下一分支的下采样

        x = self.stage4(x)
        x = self.final_layer(x[0])

        #
        # x = (0.5 * x/(1+ torch.abs(x)) + 0.5)
        # 上采样到原图
        # x = nn.Upsample(size=(h,w), mode='nearest')(x)

        # if torch.onnx.is_in_onnx_export():  # 旨在导出onnx的时候为True

        if not self.training:
            x = torch.sigmoid(x)
            h = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(x)
            keep = 1. - torch.ceil(h - x)
            # keep = (h == x).float()  # 保留下极大值点
            x = h * keep

        return x


if __name__ == '__main__':
    model = HighResolution(base_channel=32, num_joint=1)
    model.eval()
    print(model)

    x = torch.rand(1, 3, 224, 224)
    y = model(x)
    print(y)
