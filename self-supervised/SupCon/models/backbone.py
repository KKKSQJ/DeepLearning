import torchvision.models as models

# for timm models we don't have such files, since it provides a simple wrapper timm.create_model. Check tools.models.py
BACKBONES = {
    "alexnet": models.alexnet,
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
    "mobilenet_v2": models.mobilenet_v2,
    "vgg11": models.vgg11,
    "vgg11_bn": models.vgg11_bn,
    "vgg13": models.vgg13,
    "vgg13_bn": models.vgg13_bn,
    "vgg16": models.vgg16,
    "vgg16_bn": models.vgg16_bn,
    "vgg19": models.vgg19,
    "vgg19_bn": models.vgg19_bn,
    "densenet121": models.densenet121,
    "densenet169": models.densenet169,
    "densenet161": models.densenet161,
    "densenet201": models.densenet201,
    "inception_v3": models.inception_v3,
    "resnext50_32x4d": models.resnext50_32x4d,
    "resnext101_32x8d": models.resnext101_32x8d,
    "wide_resnet50": models.wide_resnet50_2,
    "wide_resnet101": models.wide_resnet101_2,
}
