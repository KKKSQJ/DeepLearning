import os
from pathlib import Path
import logging
import torch

from models.deeplabv3 import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenetv3_large

MODEL = {
    "deeplabv3_resnet50": deeplabv3_resnet50,
    "deeplabv3_resnet101": deeplabv3_resnet101,
    "deeplabv3_mobilenetv3_large": deeplabv3_mobilenetv3_large
}


def build_model(cfg='config/example.yaml', num_classes=21, pretrain=True):
    if isinstance(cfg, dict):
        config = cfg
    else:
        import yaml
        yaml_file = Path(cfg).name
        with open(cfg) as f:
            config = yaml.safe_load(f)

    model_config = config["model"]
    model_name = model_config["name"]
    aux = model_config["use_aux"]
    pretrain_backbone = model_config["pretrain_backbone"]
    pretrain_weights = model_config["pretrain_weights"]

    # num_classes = config["train"]["num_classes"]

    assert model_name in MODEL.keys(), f"model name '{model_name}' must be {MODEL.keys()}"
    model = MODEL[model_name](aux, num_classes, pretrain_backbone, pretrain_weights)

    if pretrain:
        assert os.path.exists(pretrain_weights), f"{pretrain_weights} does not exists!"
        logging.info(f"====>  Load pretrin weights from: {pretrain_weights}   <=====")
        weights_dict = torch.load(pretrain_weights, map_location='cpu')

        if num_classes != 21:
            # 官方提供的预训练权重是21类(包括背景)
            # 如果训练自己的数据集，将和类别相关的权重删除，防止权重shape不一致报错
            for k in list(weights_dict.keys()):
                if "classifier.4" in k:
                    del weights_dict[k]

        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            logging.info(f"====> Missing_keys: {missing_keys}  <====")
            logging.info(f"====> Unexpected_keys: {unexpected_keys}  <====")
    return model


if __name__ == '__main__':
    x = torch.randn(2, 3, 480, 480)
    model = build_model(cfg='../config/example.yaml', pretrain=False)
    y = model(x)
    print(model)
