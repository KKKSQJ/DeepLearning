import random

import torch
import torch.nn as nn
from models.resnet import Bottleneck, resnet50


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class BatchDrop(nn.Module):
    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h - rh)
            sy = random.randint(0, w - rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx + rh, sy:sy + rw] = 0
            x = x * mask
        return x


class BFE(nn.Module):
    def __init__(self,
                 num_classes=80,
                 stride=1,
                 width_ratio=0.5,
                 height_ratio=0.5,
                 global_feature_dim=512,
                 part_feature_dim=1024):
        super(BFE, self).__init__()
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # res_conv2
            resnet.layer2,  # res_conv3
            resnet.layer3,  # res_conv4
        )
        self.layer4 = nn.Sequential(
            Bottleneck(inplanes=1024, planes=512, stride=stride, downsample=nn.Sequential(
                nn.Conv2d(1024, 2048, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(2048),
            )),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512),
        )
        self.layer4.load_state_dict(resnet.layer4.state_dict())

        # global branch
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_reduction = nn.Sequential(
            nn.Conv2d(512 * 4, global_feature_dim, 1),
            nn.BatchNorm2d(global_feature_dim),
            nn.ReLU(inplace=True)
        )
        self.global_softmax = nn.Linear(global_feature_dim, num_classes)
        self.global_softmax.apply(weights_init_kaiming)
        self.global_reduction.apply(weights_init_kaiming)

        # part branch
        self.bottleneck = Bottleneck(2048, 512)
        self.part_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.batch_crop = BatchDrop(height_ratio, width_ratio)
        self.part_reduction = nn.Sequential(
            nn.Conv2d(2048, part_feature_dim, 1),
            nn.BatchNorm2d(part_feature_dim),
            nn.ReLU(inplace=True)
        )
        self.part_reduction.apply(weights_init_kaiming)
        self.part_softmax = nn.Linear(part_feature_dim, num_classes)
        self.part_softmax.apply(weights_init_kaiming)

    def forward(self, x):
        """
        :param x: input image tensor of (N, C, H, W)
        :return: (prediction, triplet_losses, softmax_losses)
        """
        # backbone
        x = self.backbone(x)
        # no down-sampling
        x = self.layer4(x)

        predict = []
        triplet_features = []
        softmax_features = []

        # global branch
        glob = self.global_avgpool(x)
        global_triplet_feature = self.global_reduction(glob).squeeze()
        global_softmax_class = self.global_softmax(global_triplet_feature)
        softmax_features.append(global_softmax_class)
        triplet_features.append(global_triplet_feature)
        predict.append(global_triplet_feature)

        # part branch
        x = self.bottleneck(x)
        x = self.batch_crop(x)
        part_feature = self.part_maxpool(x)
        part_triplet_feature = self.part_reduction(part_feature).squeeze()
        part_softmax_class = self.part_softmax(part_triplet_feature)
        softmax_features.append(part_softmax_class)
        triplet_features.append(part_triplet_feature)
        predict.append(part_triplet_feature)

        if self.training:
            return triplet_features, softmax_features
        else:
            return torch.cat(predict, -1)


def Init_model(name, args):
    if name == 'bfe':
        model = BFE(*args)
    elif name == 'ide':
        model = None
    elif name == 'resnet':
        model = None
    else:
        return
    return model


if __name__ == '__main__':
    from torchsummary import summary

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    net = Init_model('bfe', [80, 1, 0.3, 0.3])
    net = net.to(device)
    # summary(net, input_size=(3, 224, 224))

    net.eval()
    y = net.forward(torch.randn(3, 3, 224, 224).to(device))
    print(y)
