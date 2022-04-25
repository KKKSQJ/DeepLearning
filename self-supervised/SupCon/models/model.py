import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from models.backbone import *


def create_encoder(backbone):
    try:
        if 'timm_' in backbone:
            # backbone = backbone.split('_')[-1]
            backbone = backbone[5:]
            model = timm.create_model(model_name=backbone, pretrained=True)
        else:
            model = BACKBONES[backbone](pretrained=True)
    except RuntimeError or KeyError:
        raise RuntimeError('Specify the correct backbone name. Either one of torchvision backbones, or a timm backbone.'
                           'For timm - add prefix \'timm_\'. For instance, timm_resnet18')

    layers = torch.nn.Sequential(*list(model.children()))
    try:
        potential_last_layer = layers[-1]
        while not isinstance(potential_last_layer, nn.Linear):
            potential_last_layer = potential_last_layer[-1]
    except TypeError:
        raise TypeError('Can\'t find the linear layer of the model')

    features_dim = potential_last_layer.in_features
    model = torch.nn.Sequential(*list(model.children())[:-1])

    # encoder:除去分类层  # encoder输出的特征维度
    return model, features_dim


class SupConModel(nn.Module):
    def __init__(self, backbone='resnet50', projection_dim=128, second_stage=False, num_classes=1000):
        super(SupConModel, self).__init__()
        self.encoder, self.features_dim = create_encoder(backbone)
        self.second_stage = second_stage
        self.projection_head = True
        self.projection_dim = projection_dim
        self.embed_dim = projection_dim

        if self.second_stage:
            for param in self.encoder.parameters():
                param.requires_grad = False
            # 分类
            self.classifier = nn.Linear(self.features_dim, num_classes)
        else:
            self.head = nn.Sequential(
                nn.Linear(self.features_dim, self.features_dim),
                nn.ReLU(inplace=True),
                # 度量学习
                nn.Linear(self.features_dim, self.projection_dim))

    def use_projection_head(self, mode):
        self.projection_head = mode
        if mode:
            self.embed_dim = self.projection_dim
        else:
            self.embed_dim = self.features_dim

    def forward(self, x):
        if self.second_stage:
            feat = self.encoder(x).squeeze()
            return self.classifier(feat)
        else:
            feat = self.encoder(x).squeeze()
            if self.projection_head:
                return F.normalize(self.head(feat), dim=1)
            else:
                return F.normalize(feat, dim=1)


def build_model(backbone, second_stage=False, num_classes=None, ckpt_pretrained=None):
    model = SupConModel(backbone=backbone, second_stage=second_stage, num_classes=num_classes)

    if ckpt_pretrained:
        model.load_state_dict(torch.load(ckpt_pretrained)['model_state_dict'], strict=False)

    return model


if __name__ == '__main__':
    print(timm.list_models())
    # model_name = 'timm_efficientnet_b0'
    model_name = 'resnet50'
    # model, c = create_encoder(model_name)

    from torchsummary import summary

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 3, 224, 224).to(device)
    # 分类
    # model = SupConModel(backbone='resnet50', projection_dim=128, second_stage=True, num_classes=1000).to(device)
    # 度量学习
    model = SupConModel(backbone='resnet50', projection_dim=128, second_stage=False, num_classes=None).to(device)
    y = model(x)
    print(model)
    print(y.shape)

    summary(model, input_size=(3, 224, 224), device=device.type)
