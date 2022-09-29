from .resnest import ResNeSt, Bottleneck


# resnest
def build_model(config):
    model_type = config.MODEL.TYPE

    if model_type == 'resnest':
        model = ResNeSt(block=Bottleneck,
                        layers=config.MODEL.RESNEST.LAYERS,
                        radix=config.MODEL.RESNEST.RADIX,
                        groups=config.MODEL.RESNEST.GROUPS,
                        bottleneck_width=config.MODEL.RESNEST.BOTTLENECK_WIDTH,
                        num_classes=config.MODEL.NUM_CLASSES,
                        deep_stem=config.MODEL.RESNEST.DEEP_STEM,
                        stem_width=config.MODEL.RESNEST.STEM_WIDTH,
                        avg_down=config.MODEL.RESNEST.AVG_DOWN,
                        avd=config.MODEL.RESNEST.AVD,
                        avd_first=config.MODEL.RESNEST.AVD_FIRST)


    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
