from .resNext import ResNet, Bottleneck


# 'resnext50_32x4d', 'resnext101_32x8d'
#  'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
#  'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
def build_model(config):
    model_type = config.MODEL.TYPE

    if model_type == 'resnext':
        model = ResNet(block=Bottleneck,
                       layers=config.MODEL.RESNEXT.LAYERS,
                       num_classes=config.MODEL.NUM_CLASSES,
                       groups=config.MODEL.RESNEXT.GROUPS,
                       width_per_group=config.MODEL.RESNEXT.WIDTH_PER_GROUP)

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
