import os.path

import tensorflow.compat.v1 as tf
import torch
from torchvision.models import resnet50

tf.disable_eager_execution()
import numpy as np


def tr(v):
    # tensorflow weights to pytorch weights
    if v.ndim == 4:
        return np.ascontiguousarray(v.transpose(3, 2, 0, 1))
    elif v.ndim == 2:
        return np.ascontiguousarray(v.transpose())
    return v


def read_ckpt(ckpt):
    # https://github.com/tensorflow/tensorflow/issues/1823
    reader = tf.train.NewCheckpointReader(ckpt)
    weights = {n: reader.get_tensor(n) for (n, _) in reader.get_variable_to_shape_map().items()}
    pyweights = {k: tr(v) for (k, v) in weights.items()}
    return pyweights


def generate_trans_dict(weights):
    weights_dict = dict()
    trans_dict = {
        "model/gc-read-pyramid/conv1/weights": "pyramid_encoder.conv1.0.weight",
        "model/gc-read-pyramid/conv1/biases": "pyramid_encoder.conv1.0.bias",
        "model/gc-read-pyramid/conv2/weights": "pyramid_encoder.conv2.0.weight",
        "model/gc-read-pyramid/conv2/biases": "pyramid_encoder.conv2.0.bias",
        "model/gc-read-pyramid/conv3/weights": "pyramid_encoder.conv3.0.weight",
        "model/gc-read-pyramid/conv3/biases": "pyramid_encoder.conv3.0.bias",
        "model/gc-read-pyramid/conv4/weights": "pyramid_encoder.conv4.0.weight",
        "model/gc-read-pyramid/conv4/biases": "pyramid_encoder.conv4.0.bias",
        "model/gc-read-pyramid/conv5/weights": "pyramid_encoder.conv5.0.weight",
        "model/gc-read-pyramid/conv5/biases": "pyramid_encoder.conv5.0.bias",
        "model/gc-read-pyramid/conv6/weights": "pyramid_encoder.conv6.0.weight",
        "model/gc-read-pyramid/conv6/biases": "pyramid_encoder.conv6.0.bias",
        "model/gc-read-pyramid/conv7/weights": "pyramid_encoder.conv7.0.weight",
        "model/gc-read-pyramid/conv7/biases": "pyramid_encoder.conv7.0.bias",
        "model/gc-read-pyramid/conv8/weights": "pyramid_encoder.conv8.0.weight",
        "model/gc-read-pyramid/conv8/biases": "pyramid_encoder.conv8.0.bias",
        "model/gc-read-pyramid/conv9/weights": "pyramid_encoder.conv9.0.weight",
        "model/gc-read-pyramid/conv9/biases": "pyramid_encoder.conv9.0.bias",
        "model/gc-read-pyramid/conv10/weights": "pyramid_encoder.conv10.0.weight",
        "model/gc-read-pyramid/conv10/biases": "pyramid_encoder.conv10.0.bias",
        "model/gc-read-pyramid/conv11/weights": "pyramid_encoder.conv11.0.weight",
        "model/gc-read-pyramid/conv11/biases": "pyramid_encoder.conv11.0.bias",
        "model/gc-read-pyramid/conv12/weights": "pyramid_encoder.conv12.0.weight",
        "model/gc-read-pyramid/conv12/biases": "pyramid_encoder.conv12.0.bias",
        "model/G6/fgc-volume-filtering-6/disp-1/weights": "disparity_decoder_6.decoder.fgc-volume-filtering/disp1.0.weight",
        "model/G6/fgc-volume-filtering-6/disp-1/biases": "disparity_decoder_6.decoder.fgc-volume-filtering/disp1.0.bias",
        "model/G6/fgc-volume-filtering-6/disp-2/weights": "disparity_decoder_6.decoder.fgc-volume-filtering/disp2.0.weight",
        "model/G6/fgc-volume-filtering-6/disp-2/biases": "disparity_decoder_6.decoder.fgc-volume-filtering/disp2.0.bias",
        "model/G6/fgc-volume-filtering-6/disp-3/weights": "disparity_decoder_6.decoder.fgc-volume-filtering/disp3.0.weight",
        "model/G6/fgc-volume-filtering-6/disp-3/biases": "disparity_decoder_6.decoder.fgc-volume-filtering/disp3.0.bias",
        "model/G6/fgc-volume-filtering-6/disp-4/weights": "disparity_decoder_6.decoder.fgc-volume-filtering/disp4.0.weight",
        "model/G6/fgc-volume-filtering-6/disp-4/biases": "disparity_decoder_6.decoder.fgc-volume-filtering/disp4.0.bias",
        "model/G6/fgc-volume-filtering-6/disp-5/weights": "disparity_decoder_6.decoder.fgc-volume-filtering/disp5.0.weight",
        "model/G6/fgc-volume-filtering-6/disp-5/biases": "disparity_decoder_6.decoder.fgc-volume-filtering/disp5.0.bias",
        "model/G6/fgc-volume-filtering-6/disp-6/weights": "disparity_decoder_6.decoder.fgc-volume-filtering/disp6.0.weight",
        "model/G6/fgc-volume-filtering-6/disp-6/biases": "disparity_decoder_6.decoder.fgc-volume-filtering/disp6.0.bias",
        "model/G5/fgc-volume-filtering-5/disp-1/weights": "disparity_decoder_5.decoder.fgc-volume-filtering/disp1.0.weight",
        "model/G5/fgc-volume-filtering-5/disp-1/biases": "disparity_decoder_5.decoder.fgc-volume-filtering/disp1.0.bias",
        "model/G5/fgc-volume-filtering-5/disp-2/weights": "disparity_decoder_5.decoder.fgc-volume-filtering/disp2.0.weight",
        "model/G5/fgc-volume-filtering-5/disp-2/biases": "disparity_decoder_5.decoder.fgc-volume-filtering/disp2.0.bias",
        "model/G5/fgc-volume-filtering-5/disp-3/weights": "disparity_decoder_5.decoder.fgc-volume-filtering/disp3.0.weight",
        "model/G5/fgc-volume-filtering-5/disp-3/biases": "disparity_decoder_5.decoder.fgc-volume-filtering/disp3.0.bias",
        "model/G5/fgc-volume-filtering-5/disp-4/weights": "disparity_decoder_5.decoder.fgc-volume-filtering/disp4.0.weight",
        "model/G5/fgc-volume-filtering-5/disp-4/biases": "disparity_decoder_5.decoder.fgc-volume-filtering/disp4.0.bias",
        "model/G5/fgc-volume-filtering-5/disp-5/weights": "disparity_decoder_5.decoder.fgc-volume-filtering/disp5.0.weight",
        "model/G5/fgc-volume-filtering-5/disp-5/biases": "disparity_decoder_5.decoder.fgc-volume-filtering/disp5.0.bias",
        "model/G5/fgc-volume-filtering-5/disp-6/weights": "disparity_decoder_5.decoder.fgc-volume-filtering/disp6.0.weight",
        "model/G5/fgc-volume-filtering-5/disp-6/biases": "disparity_decoder_5.decoder.fgc-volume-filtering/disp6.0.bias",
        "model/G4/fgc-volume-filtering-4/disp-1/weights": "disparity_decoder_4.decoder.fgc-volume-filtering/disp1.0.weight",
        "model/G4/fgc-volume-filtering-4/disp-1/biases": "disparity_decoder_4.decoder.fgc-volume-filtering/disp1.0.bias",
        "model/G4/fgc-volume-filtering-4/disp-2/weights": "disparity_decoder_4.decoder.fgc-volume-filtering/disp2.0.weight",
        "model/G4/fgc-volume-filtering-4/disp-2/biases": "disparity_decoder_4.decoder.fgc-volume-filtering/disp2.0.bias",
        "model/G4/fgc-volume-filtering-4/disp-3/weights": "disparity_decoder_4.decoder.fgc-volume-filtering/disp3.0.weight",
        "model/G4/fgc-volume-filtering-4/disp-3/biases": "disparity_decoder_4.decoder.fgc-volume-filtering/disp3.0.bias",
        "model/G4/fgc-volume-filtering-4/disp-4/weights": "disparity_decoder_4.decoder.fgc-volume-filtering/disp4.0.weight",
        "model/G4/fgc-volume-filtering-4/disp-4/biases": "disparity_decoder_4.decoder.fgc-volume-filtering/disp4.0.bias",
        "model/G4/fgc-volume-filtering-4/disp-5/weights": "disparity_decoder_4.decoder.fgc-volume-filtering/disp5.0.weight",
        "model/G4/fgc-volume-filtering-4/disp-5/biases": "disparity_decoder_4.decoder.fgc-volume-filtering/disp5.0.bias",
        "model/G4/fgc-volume-filtering-4/disp-6/weights": "disparity_decoder_4.decoder.fgc-volume-filtering/disp6.0.weight",
        "model/G4/fgc-volume-filtering-4/disp-6/biases": "disparity_decoder_4.decoder.fgc-volume-filtering/disp6.0.bias",
        "model/G3/fgc-volume-filtering-3/disp-1/weights": "disparity_decoder_3.decoder.fgc-volume-filtering/disp1.0.weight",
        "model/G3/fgc-volume-filtering-3/disp-1/biases": "disparity_decoder_3.decoder.fgc-volume-filtering/disp1.0.bias",
        "model/G3/fgc-volume-filtering-3/disp-2/weights": "disparity_decoder_3.decoder.fgc-volume-filtering/disp2.0.weight",
        "model/G3/fgc-volume-filtering-3/disp-2/biases": "disparity_decoder_3.decoder.fgc-volume-filtering/disp2.0.bias",
        "model/G3/fgc-volume-filtering-3/disp-3/weights": "disparity_decoder_3.decoder.fgc-volume-filtering/disp3.0.weight",
        "model/G3/fgc-volume-filtering-3/disp-3/biases": "disparity_decoder_3.decoder.fgc-volume-filtering/disp3.0.bias",
        "model/G3/fgc-volume-filtering-3/disp-4/weights": "disparity_decoder_3.decoder.fgc-volume-filtering/disp4.0.weight",
        "model/G3/fgc-volume-filtering-3/disp-4/biases": "disparity_decoder_3.decoder.fgc-volume-filtering/disp4.0.bias",
        "model/G3/fgc-volume-filtering-3/disp-5/weights": "disparity_decoder_3.decoder.fgc-volume-filtering/disp5.0.weight",
        "model/G3/fgc-volume-filtering-3/disp-5/biases": "disparity_decoder_3.decoder.fgc-volume-filtering/disp5.0.bias",
        "model/G3/fgc-volume-filtering-3/disp-6/weights": "disparity_decoder_3.decoder.fgc-volume-filtering/disp6.0.weight",
        "model/G3/fgc-volume-filtering-3/disp-6/biases": "disparity_decoder_3.decoder.fgc-volume-filtering/disp6.0.bias",
        "model/G2/fgc-volume-filtering-2/disp-1/weights": "disparity_decoder_2.decoder.fgc-volume-filtering/disp1.0.weight",
        "model/G2/fgc-volume-filtering-2/disp-1/biases": "disparity_decoder_2.decoder.fgc-volume-filtering/disp1.0.bias",
        "model/G2/fgc-volume-filtering-2/disp-2/weights": "disparity_decoder_2.decoder.fgc-volume-filtering/disp2.0.weight",
        "model/G2/fgc-volume-filtering-2/disp-2/biases": "disparity_decoder_2.decoder.fgc-volume-filtering/disp2.0.bias",
        "model/G2/fgc-volume-filtering-2/disp-3/weights": "disparity_decoder_2.decoder.fgc-volume-filtering/disp3.0.weight",
        "model/G2/fgc-volume-filtering-2/disp-3/biases": "disparity_decoder_2.decoder.fgc-volume-filtering/disp3.0.bias",
        "model/G2/fgc-volume-filtering-2/disp-4/weights": "disparity_decoder_2.decoder.fgc-volume-filtering/disp4.0.weight",
        "model/G2/fgc-volume-filtering-2/disp-4/biases": "disparity_decoder_2.decoder.fgc-volume-filtering/disp4.0.bias",
        "model/G2/fgc-volume-filtering-2/disp-5/weights": "disparity_decoder_2.decoder.fgc-volume-filtering/disp5.0.weight",
        "model/G2/fgc-volume-filtering-2/disp-5/biases": "disparity_decoder_2.decoder.fgc-volume-filtering/disp5.0.bias",
        "model/G2/fgc-volume-filtering-2/disp-6/weights": "disparity_decoder_2.decoder.fgc-volume-filtering/disp6.0.weight",
        "model/G2/fgc-volume-filtering-2/disp-6/biases": "disparity_decoder_2.decoder.fgc-volume-filtering/disp6.0.bias",
        "model/context-1/weights": "refinement_module.context1.0.weight",
        "model/context-1/biases": "refinement_module.context1.0.bias",
        "model/context-2/weights": "refinement_module.context2.0.weight",
        "model/context-2/biases": "refinement_module.context2.0.bias",
        "model/context-3/weights": "refinement_module.context3.0.weight",
        "model/context-3/biases": "refinement_module.context3.0.bias",
        "model/context-4/weights": "refinement_module.context4.0.weight",
        "model/context-4/biases": "refinement_module.context4.0.bias",
        "model/context-5/weights": "refinement_module.context5.0.weight",
        "model/context-5/biases": "refinement_module.context5.0.bias",
        "model/context-6/weights": "refinement_module.context6.0.weight",
        "model/context-6/biases": "refinement_module.context6.0.bias",
        "model/context-7/weights": "refinement_module.context7.0.weight",
        "model/context-7/biases": "refinement_module.context7.0.bias",
    }
    for name, data in weights.items():
        assert name in trans_dict.keys()
        torch_name = trans_dict[name]
        weights_dict[torch_name] = data

    return weights_dict


def save_py_weights(weights_dict, save_path):
    os.makedirs(save_path,exist_ok=True)
    for k, v in weights_dict.items():
        weights_dict[k] = torch.as_tensor(v)

    torch.save(weights_dict, os.path.join(save_path, "model.pth"))
    print("Conversion complete.")


def load_py_weights(weight_path):
    torch_weights_dict = torch.load(weight_path, map_location='cpu')
    return torch_weights_dict


def read_py_model(name=None):
    model = resnet50()
    py_keys = list(model.named_parameters())
    return py_keys


if __name__ == '__main__':
    # import torch
    #
    # a = torch.tensor([0, 0, 2, 4])  # .resize(2,2)
    # a1 = a.clone()
    # b = torch.tensor([0, 1, 0, 1])
    # a1[b == 1] = 1
    # a1[a == 4] = 4
    #
    # print(1)

    ckpt = 'MADNet/synthetic/weights.ckpt'
    pyweights = read_ckpt(ckpt)
    weights = generate_trans_dict(pyweights)
    save_py_weights(weights, "out")
    print("ok")
