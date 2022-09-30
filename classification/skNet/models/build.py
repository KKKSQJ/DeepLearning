from .sknet import SKBlock, SKNet


# SKNET
def build_model(config):
    model_type = config.MODEL.TYPE

    if model_type == 'sknet':
        model = SKNet(block=SKBlock,
                      layers=config.MODEL.SKNET.LAYERS,
                      num_classes=config.MODEL.NUM_CLASSES,
                      M=config.MODEL.SKNET.M,
                      G=config.MODEL.SKNET.G,
                      r=config.MODEL.SKNET.R,
                      stem_width=config.MODEL.SKNET.STEM_WIDTH,
                      deep_stem=config.MODEL.SKNET.DEEP_STEM)



    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
