import os

from models.seg_hrnet import HighResulutionNet
import torch

MODEL = {"seg_hrnet": HighResulutionNet,
         "seg_hrnet_ocr": 1}


def build_model(in_channel=None, num_classes=None, config=None):
    if isinstance(config, dict):
        config = config
    else:
        import yaml
        with open(config) as f:
            config = yaml.safe_load(f)  # model dict
    model_name = config["model"]["name"]
    ckpt_pretrained = config["model"]["pretrained"]

    assert model_name in MODEL.keys(), f"model name must be {MODEL.keys()}"

    model = MODEL[model_name](in_channel=in_channel, num_classes=num_classes, cfg=config)

    if ckpt_pretrained:
        assert os.path.exists(ckpt_pretrained), f"{ckpt_pretrained} dose not exists"
        pretrained_dict = torch.load(ckpt_pretrained)

        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict["state_dict"].items() if k in model_dict.keys()}
        for k, _ in pretrained_dict.items():
            # logger.info(
            #     '=> loading {} pretrained model {}'.format(k, ckpt_pretrained))
            print(
                '=> loading {} pretrained model {}'.format(k, ckpt_pretrained))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


if __name__ == '__main__':
    model = build_model(config='../config/example.yaml')
