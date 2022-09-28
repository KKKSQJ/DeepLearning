from .googlenet import GoogLeNet


# googlenet
# https://download.pytorch.org/models/googlenet-1378be20.pth
def build_model(config):
    model_type = config.MODEL.TYPE

    if model_type == 'googlenet':
        model = GoogLeNet(num_classes=config.MODEL.NUM_CLASSES,
                          aux_logits=config.MODEL.GOOGLENET.AUX_LOGITS,
                          transform_input=config.MODEL.GOOGLENET.TRANSFORM_INPUTS,
                          init_weights=config.MODEL.GOOGLENET.INIT_WEIGHTS)

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model


