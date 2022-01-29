import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from collections import OrderedDict

__all__ = ["coatnet_0", "coatnet_1", "coatnet_2", "coatnet_3", "coatnet_4"]


def conv_3x3_bn(in_c, out_c, image_size, downsample=False):
    stride = 2 if downsample else 1
    layer = nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.GELU()
    )
    return layer


class SE(nn.Module):
    def __init__(self, in_c, out_c, expansion=0.25):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_c, int(in_c * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(in_c * expansion), out_c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MBConv(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 image_size,
                 downsample=False,
                 expansion=4):
        super(MBConv, self).__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1
        hidden_dim = int(in_c * expansion)

        if self.downsample:
            # 只有第一层的时候，进行下采样
            # self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
            # self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # self.proj = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)

            self.downsample_layer = nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)
            )

        layers = OrderedDict()
        # expand
        expand_conv = nn.Sequential(
            nn.Conv2d(in_c, hidden_dim, 1, stride, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        layers.update({"expand_conv": expand_conv})

        # Depwise Conv
        dw_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        layers.update({"dw_conv": dw_conv})

        # se
        layers.update({"se": SE(in_c, hidden_dim)})

        # project
        pro_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_c)
        )
        layers.update({"pro_conv": pro_conv})
        self.block = nn.Sequential(layers)

    def forward(self, x):
        if self.downsample:
            return self.downsample_layer(x) + self.block(x)
        else:
            return x + self.block(x)


class Attention(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 image_size,
                 heads=8,
                 dim_head=32,
                 dropout=0.):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == in_c)

        self.ih, self.iw = image_size if len(image_size) == 2 else (image_size, image_size)

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads)
        )

        coords = torch.meshgrid((torch.arange(self.ih), torch.arange(self.iw)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]

        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)

        """
        PyTorch中定义模型时，self.register_buffer('name', Tensor)，
        该方法的作用是定义一组参数，该组参数的特别之处在于：
        模型训练时不会更新（即调用 optimizer.step() 后该组参数不会变化，只可人为地改变它们的值），
        但是保存模型时，该组参数又作为模型参数不可或缺的一部分被保存。
        """
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)
        self.qkv = nn.Linear(in_c, inner_dim * 3, bias=False)
        self.proj = nn.Sequential(
            nn.Linear(inner_dim, out_c),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # [q,k,v]
        qkv = self.qkv(x).chunk(3, dim=-1)
        # q,k,v:[batch_size, num_heads, num_patches, head_dim]
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # [batch_size, num_heads, ih*iw, ih*iw]
        # 时间复杂度：O(图片边长的平方)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads)
        )
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih * self.iw, w=self.ih * self.iw
        )
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        return out


class FFN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FFN, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)


class Transformer(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 image_size,
                 heads=8,
                 dim_head=32,
                 downsample=False,
                 dropout=0.,
                 expansion=4,
                 norm_layer=nn.LayerNorm):
        super(Transformer, self).__init__()
        self.downsample = downsample
        hidden_dim = int(in_c * expansion)
        self.ih, self.iw = image_size

        if self.downsample:
            # 第一层进行下采样
            self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.proj = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)

        self.attn = Attention(in_c, out_c, image_size, heads, dim_head, dropout)
        self.ffn = FFN(out_c, hidden_dim)
        self.norm1 = norm_layer(in_c)
        self.norm2 = norm_layer(out_c)

    def forward(self, x):
        x1 = self.pool1(x) if self.downsample else x
        x1 = rearrange(x1, 'b c h w -> b (h w) c')
        x1 = self.attn(self.norm1(x1))
        x1 = rearrange(x1, 'b (h w) c -> b c h w', h=self.ih, w=self.iw)
        x2 = self.proj((self.pool2(x))) if self.downsample else x

        x3 = x1 + x2
        x4 = rearrange(x3, 'b c h w -> b (h w) c')
        x4 = self.ffn(self.norm2(x4))
        x4 = rearrange(x4, 'b (h w) c -> b c h w', h=self.ih, w=self.iw)
        out = x3 + x4
        return out


class CoAtNet(nn.Module):
    def __init__(self,
                 image_size=(224, 224),
                 in_channels: int = 3,
                 num_blocks: list = [2, 2, 3, 5, 2],  # L
                 channels: list = [64, 96, 192, 384, 768],  # D
                 num_classes: int = 1000,
                 block_types=['C', 'C', 'T', 'T']):
        super(CoAtNet, self).__init__()

        assert len(image_size) == 2, "image size must be: {H,W}"
        assert len(channels) == 5
        assert len(block_types) == 4

        ih, iw = image_size
        block = {'C': MBConv, 'T': Transformer}

        self.s0 = self._make_layer(
            conv_3x3_bn, in_channels, channels[0], num_blocks[0], (ih // 2, iw // 2)
        )
        self.s1 = self._make_layer(
            block[block_types[0]], channels[0], channels[1], num_blocks[1], (ih // 4, iw // 4)
        )
        self.s2 = self._make_layer(
            block[block_types[1]], channels[1], channels[2], num_blocks[2], (ih // 8, iw // 8)
        )
        self.s3 = self._make_layer(
            block[block_types[2]], channels[2], channels[3], num_blocks[3], (ih // 16, iw // 16)
        )
        self.s4 = self._make_layer(
            block[block_types[3]], channels[3], channels[4], num_blocks[4], (ih // 32, iw // 32)
        )

        # 总共下采样32倍 2^5=32
        self.pool = nn.AvgPool2d(ih // 32, 1)
        self.fc = nn.Linear(channels[-1], num_classes, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, in_c, out_c, depth, image_size):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(block(in_c, out_c, image_size, downsample=True))
            else:
                layers.append(block(out_c, out_c, image_size, downsample=False))
        return nn.Sequential(*layers)


def coatnet_0(num_classes=1000):
    num_blocks = [2, 2, 3, 5, 2]  # L
    channels = [64, 96, 192, 384, 768]  # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=num_classes)


def coatnet_1(num_classes=1000):
    num_blocks = [2, 2, 6, 14, 2]  # L
    channels = [64, 96, 192, 384, 768]  # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=num_classes)


def coatnet_2(num_classes=1000):
    num_blocks = [2, 2, 6, 14, 2]  # L
    channels = [128, 128, 256, 512, 1026]  # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=num_classes)


def coatnet_3(num_classes=1000):
    num_blocks = [2, 2, 6, 14, 2]  # L
    channels = [192, 192, 384, 768, 1536]  # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=num_classes)


def coatnet_4(num_classes=1000):
    num_blocks = [2, 2, 12, 28, 2]  # L
    channels = [192, 192, 384, 768, 1536]  # D
    return CoAtNet((224, 224), 3, num_blocks, channels, num_classes=num_classes)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = torch.randn(1, 3, 224, 224).to(device)
    model = coatnet_0().to(device)
    out = model(img)
    summary(model, input_size=(3, 224, 224))
    print(out.shape, count_parameters(model))

    # net = coatnet_1()
    # out = net(img)
    # print(out.shape, count_parameters(net))
    #
    # net = coatnet_2()
    # out = net(img)
    # print(out.shape, count_parameters(net))
    #
    # net = coatnet_3()
    # out = net(img)
    # print(out.shape, count_parameters(net))
    #
    # net = coatnet_4()
    # out = net(img)
    # print(out.shape, count_parameters(net))
