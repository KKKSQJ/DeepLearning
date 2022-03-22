import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channel, out_channel, mid_channel=None):
        super(DoubleConv, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        if not mid_channel:
            mid_channel = out_channel

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channel, out_channel):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channel, out_channel)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channel, out_channel, bilinear=True):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channel, out_channel, in_channel // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channel, out_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input shape is BCHW
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2], mode='reflect')
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=[64, 128, 256, 512, 1024], classes=2, bilinear=False):
        super(UNet, self).__init__()
        self.in_channel = in_channel
        self.classes = classes
        self.bilinear = bilinear
        if not out_channel:
            out_channel = [64, 128, 256, 512, 1024]

        self.inc = DoubleConv(in_channel, out_channel[0])
        self.down1 = Down(out_channel[0], out_channel[1])
        self.down2 = Down(out_channel[1], out_channel[2])
        self.down3 = Down(out_channel[2], out_channel[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(out_channel[3], out_channel[4] // factor)

        self.up1 = Up(out_channel[4], out_channel[3] // factor, bilinear)
        self.up2 = Up(out_channel[3], out_channel[2] // factor, bilinear)
        self.up3 = Up(out_channel[2], out_channel[1] // factor, bilinear)
        self.up4 = Up(out_channel[1], out_channel[0] // factor, bilinear)

        self.outc = OutConv(out_channel[0] // factor, classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


if __name__ == '__main__':
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.rand(1, 3, 224, 224).to(device)
    model = UNet(bilinear=False).to(device)
    y = model(x)
    print(y)
    summary(model, (3, 224, 224))
    print(model)
