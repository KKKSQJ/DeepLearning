import torch
from torch import nn, Tensor
import torch.nn.functional as F

import numpy as np
from typing import Dict
from collections import OrderedDict

from torch.autograd import Variable

from utils.op_utils import stereo_cost_volume_correlation, gather_nd_torch, resize_image_with_crop_or_pad,resize_image_with_crop_or_pad_2
from data_utils import preprocessing


class Pyramid_Encoder(nn.Module):
    def __init__(self, input_channel=3, layer_prefix='pyramid', out_channels=None, activation=None, BN=False):
        super(Pyramid_Encoder, self).__init__()
        if out_channels is None:
            out_channels = [16, 32, 64, 96, 128, 192]
        assert len(out_channels) == 6

        if activation is None:
            activation = nn.LeakyReLU(0.2)

        names = []
        layers = OrderedDict()

        # conv1
        names.append('{}/conv1'.format(layer_prefix))
        conv1 = nn.Sequential(
            nn.Conv2d(input_channel, out_channels[0], 3, 2, 1),
            nn.BatchNorm2d(out_channels[0]) if BN else nn.Identity(),
            activation
        )
        layers.update({names[-1]: conv1})

        # conv2
        names.append('{}/conv2'.format(layer_prefix))
        conv2 = nn.Sequential(
            nn.Conv2d(out_channels[0], out_channels[0], 3, 1, 1),
            nn.BatchNorm2d(out_channels[0]) if BN else nn.Identity(),
            activation
        )
        layers.update({names[-1]: conv2})

        # conv3
        names.append('{}/conv3'.format(layer_prefix))
        conv3 = nn.Sequential(
            nn.Conv2d(out_channels[0], out_channels[1], 3, 2, 1),
            nn.BatchNorm2d(out_channels[1]) if BN else nn.Identity(),
            activation
        )
        layers.update({names[-1]: conv3})

        # conv4
        names.append('{}/conv4'.format(layer_prefix))
        conv4 = nn.Sequential(
            nn.Conv2d(out_channels[1], out_channels[1], 3, 1, 1),
            nn.BatchNorm2d(out_channels[1]) if BN else nn.Identity(),
            activation
        )
        layers.update({names[-1]: conv4})

        # conv5
        names.append('{}/conv5'.format(layer_prefix))
        conv5 = nn.Sequential(
            nn.Conv2d(out_channels[1], out_channels[2], 3, 2, 1),
            nn.BatchNorm2d(out_channels[2]) if BN else nn.Identity(),
            activation
        )
        layers.update({names[-1]: conv5})

        # conv6
        names.append('{}/conv6'.format(layer_prefix))
        conv6 = nn.Sequential(
            nn.Conv2d(out_channels[2], out_channels[2], 3, 1, 1),
            nn.BatchNorm2d(out_channels[2]) if BN else nn.Identity(),
            activation
        )
        layers.update({names[-1]: conv6})

        # conv7
        names.append('{}/conv7'.format(layer_prefix))
        conv7 = nn.Sequential(
            nn.Conv2d(out_channels[2], out_channels[3], 3, 2, 1),
            nn.BatchNorm2d(out_channels[3]) if BN else nn.Identity(),
            activation
        )
        layers.update({names[-1]: conv7})

        # conv8
        names.append('{}/conv8'.format(layer_prefix))
        conv8 = nn.Sequential(
            nn.Conv2d(out_channels[3], out_channels[3], 3, 1, 1),
            nn.BatchNorm2d(out_channels[3]) if BN else nn.Identity(),
            activation
        )
        layers.update({names[-1]: conv8})

        # conv9
        names.append('{}/conv9'.format(layer_prefix))
        conv9 = nn.Sequential(
            nn.Conv2d(out_channels[3], out_channels[4], 3, 2, 1),
            nn.BatchNorm2d(out_channels[4]) if BN else nn.Identity(),
            activation
        )
        layers.update({names[-1]: conv9})

        # conv10
        names.append('{}/conv10'.format(layer_prefix))
        conv10 = nn.Sequential(
            nn.Conv2d(out_channels[4], out_channels[4], 3, 1, 1),
            nn.BatchNorm2d(out_channels[4]) if BN else nn.Identity(),
            activation
        )
        layers.update({names[-1]: conv10})

        # conv11
        names.append('{}/conv11'.format(layer_prefix))
        conv11 = nn.Sequential(
            nn.Conv2d(out_channels[4], out_channels[5], 3, 2, 1),
            nn.BatchNorm2d(out_channels[5]) if BN else nn.Identity(),
            activation
        )
        layers.update({names[-1]: conv11})

        # conv12
        names.append('{}/conv12'.format(layer_prefix))
        conv12 = nn.Sequential(
            nn.Conv2d(out_channels[5], out_channels[5], 3, 1, 1),
            nn.BatchNorm2d(out_channels[5]) if BN else nn.Identity(),
            activation
        )
        layers.update({names[-1]: conv12})

        # self.f1 = nn.Sequential(*[conv1, conv2])
        # self.f2 = nn.Sequential(*[conv3, conv4])
        # self.f3 = nn.Sequential(*[conv5, conv6])
        # self.f4 = nn.Sequential(*[conv7, conv8])
        # self.f5 = nn.Sequential(*[conv9, conv10])
        # self.f6 = nn.Sequential(*[conv11, conv12])

        self.conv1 = conv1
        self.conv2 = conv2
        self.conv3 = conv3
        self.conv4 = conv4
        self.conv5 = conv5
        self.conv6 = conv6
        self.conv7 = conv7
        self.conv8 = conv8
        self.conv9 = conv9
        self.conv10 = conv10
        self.conv11 = conv11
        self.conv12 = conv12

    def forward(self, x):
        out = OrderedDict()
        x = self.conv1(x)
        f1 = self.conv2(x)
        x = self.conv3(f1)
        f2 = self.conv4(x)
        x = self.conv5(f2)
        f3 = self.conv6(x)
        x = self.conv7(f3)
        f4 = self.conv8(x)
        x = self.conv9(f4)
        f5 = self.conv10(x)
        x = self.conv11(f5)
        f6 = self.conv12(x)

        out['f1'] = f1
        out['f2'] = f2
        out['f3'] = f3
        out['f4'] = f4
        out['f5'] = f5
        out['f6'] = f6

        return out


class Disparity_Decoder(nn.Module):
    def __init__(self, scope='fgc-volume-filtering', in_channel=192, out_channels=None,
                 activation=None, BN=False):
        super(Disparity_Decoder, self).__init__()
        # self.costs = costs
        # if upsampled_disp is not None:
        #     volume = torch.cat([costs, upsampled_disp], 1)
        # else:
        #     volume = costs
        if out_channels is None:
            out_channels = [128, 128, 96, 64, 32, 1]
        assert len(out_channels) == 6
        if activation is None:
            activation = nn.LeakyReLU(0.2)

        names = []
        layers = OrderedDict()

        # disp-1
        names.append('{}/disp1'.format(scope))
        disp1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channels[0], 3, 1, 1),
            nn.BatchNorm2d(out_channels[0]) if BN else nn.Identity(),
            activation
        )
        layers.update({names[-1]: disp1})

        # disp-2
        names.append('{}/disp2'.format(scope))
        disp2 = nn.Sequential(
            nn.Conv2d(out_channels[0], out_channels[1], 3, 1, 1),
            nn.BatchNorm2d(out_channels[1]) if BN else nn.Identity(),
            activation
        )
        layers.update({names[-1]: disp2})

        # disp-3
        names.append('{}/disp3'.format(scope))
        disp3 = nn.Sequential(
            nn.Conv2d(out_channels[1], out_channels[2], 3, 1, 1),
            nn.BatchNorm2d(out_channels[2]) if BN else nn.Identity(),
            activation
        )
        layers.update({names[-1]: disp3})

        # disp-4
        names.append('{}/disp4'.format(scope))
        disp4 = nn.Sequential(
            nn.Conv2d(out_channels[2], out_channels[3], 3, 1, 1),
            nn.BatchNorm2d(out_channels[3]) if BN else nn.Identity(),
            activation
        )
        layers.update({names[-1]: disp4})

        # disp-5
        names.append('{}/disp5'.format(scope))
        disp5 = nn.Sequential(
            nn.Conv2d(out_channels[3], out_channels[4], 3, 1, 1),
            nn.BatchNorm2d(out_channels[4]) if BN else nn.Identity(),
            activation
        )
        layers.update({names[-1]: disp5})

        # disp-6
        names.append('{}/disp6'.format(scope))
        disp6 = nn.Sequential(
            nn.Conv2d(out_channels[4], out_channels[5], 3, 1, 1),
            nn.BatchNorm2d(out_channels[5]) if BN else nn.Identity(),
            nn.Identity()
        )
        layers.update({names[-1]: disp6})

        self.decoder = nn.Sequential(layers)

    def forward(self, x):
        out = self.decoder(x)
        return out


class Refinement_Module(nn.Module):
    def __init__(self, in_channel=33, out_channel=None, dilation_rate=None, activation=None, BN=False):
        super(Refinement_Module, self).__init__()
        if activation is None:
            activation = nn.LeakyReLU(0.2)
        if out_channel is None:
            out_channel = [128, 128, 128, 96, 64, 32, 1]
        assert len(out_channel) == 7
        if dilation_rate is None:
            dilation_rate = [1, 2, 4, 8, 16, 1, 1]
        assert len(dilation_rate) == 7

        names = []
        layers = OrderedDict()
        self.context1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel[0], 3, 1, dilation_rate[0], dilation=dilation_rate[0]),
            nn.BatchNorm2d(out_channel[0]) if BN else nn.Identity(),
            activation
        )

        self.context2 = nn.Sequential(
            nn.Conv2d(out_channel[0], out_channel[1], 3, 1, dilation_rate[1], dilation=dilation_rate[1]),
            nn.BatchNorm2d(out_channel[1]) if BN else nn.Identity(),
            activation
        )

        self.context3 = nn.Sequential(
            nn.Conv2d(out_channel[1], out_channel[2], 3, 1, dilation_rate[2], dilation=dilation_rate[2]),
            nn.BatchNorm2d(out_channel[2]) if BN else nn.Identity(),
            activation
        )

        self.context4 = nn.Sequential(
            nn.Conv2d(out_channel[2], out_channel[3], 3, 1, dilation_rate[3], dilation=dilation_rate[3]),
            nn.BatchNorm2d(out_channel[3]) if BN else nn.Identity(),
            activation
        )
        self.context5 = nn.Sequential(
            nn.Conv2d(out_channel[3], out_channel[4], 3, 1, dilation_rate[4], dilation=dilation_rate[4]),
            nn.BatchNorm2d(out_channel[4]) if BN else nn.Identity(),
            activation
        )
        self.context6 = nn.Sequential(
            nn.Conv2d(out_channel[4], out_channel[5], 3, 1, dilation_rate[5], dilation=dilation_rate[5]),
            nn.BatchNorm2d(out_channel[5]) if BN else nn.Identity(),
            activation
        )
        self.context7 = nn.Sequential(
            nn.Conv2d(out_channel[5], out_channel[6], 3, 1, dilation_rate[6], dilation=dilation_rate[6]),
            nn.BatchNorm2d(out_channel[6]) if BN else nn.Identity(),
            nn.Identity()
        )

    def forward(self, x):
        x = self.context1(x)
        x = self.context2(x)
        x = self.context3(x)
        x = self.context4(x)
        x = self.context5(x)
        x = self.context6(x)
        x = self.context7(x)
        return x


class MadNet(nn.Module):
    def __init__(self, pyramid_encoder, disparity_decoder, refinement_module, args={}):
        """
        Creation of a MadNet for stereo prediction
        """
        super(MadNet, self).__init__()
        self.natName = "MADNet"
        self.args = args
        self.scales = [1, 2, 4, 8, 16, 32, 64]
        self.encoder_out_channels = [16, 32, 64, 96, 128, 192]
        self.decoder_out_channels = [128, 128, 96, 64, 32, 1]
        self.refinement_out_channles = [128, 128, 128, 96, 64, 32, 1]
        self.dilation_rate = [1, 2, 4, 8, 16, 1, 1]
        self.relu = nn.ReLU(inplace=True)
        self._check_before_run(self.args)

        # pyramid_encoder
        self.pyramid_encoder = pyramid_encoder(input_channel=3, out_channels=self.encoder_out_channels)

        # disparity decoder
        self.disparity_decoder_6 = disparity_decoder(
            in_channel=2 * self.args['radius_x'] + self.args['stride'] + self.encoder_out_channels[-1],
            out_channels=self.decoder_out_channels)
        self.disparity_decoder_5 = disparity_decoder(
            in_channel=2 * self.args['radius_x'] + self.args['stride'] + self.encoder_out_channels[-2] +
                       self.decoder_out_channels[-1],
            out_channels=self.decoder_out_channels)
        self.disparity_decoder_4 = disparity_decoder(
            in_channel=2 * self.args['radius_x'] + self.args['stride'] + self.encoder_out_channels[-3] +
                       self.decoder_out_channels[-1],
            out_channels=self.decoder_out_channels)
        self.disparity_decoder_3 = disparity_decoder(
            in_channel=2 * self.args['radius_x'] + self.args['stride'] + self.encoder_out_channels[-4] +
                       self.decoder_out_channels[-1],
            out_channels=self.decoder_out_channels)
        self.disparity_decoder_2 = disparity_decoder(
            in_channel=2 * self.args['radius_x'] + self.args['stride'] + self.encoder_out_channels[-5] +
                       self.decoder_out_channels[-1],
            out_channels=self.decoder_out_channels)

        # refinement_module
        self.refinement_module = refinement_module(in_channel=33, out_channel=self.refinement_out_channles,
                                                   dilation_rate=self.dilation_rate, BN=False)

        # 初始化权重
        self.apply(self._init_weights)

    def forward(self, left_input_batch, right_input_batch):
        _disparity = []
        self.restore_shape = left_input_batch.shape[2:4]

        left_input_batch = preprocessing.pad_image(left_input_batch, 64)
        right_input_batch = preprocessing.pad_image(right_input_batch, 64)

        self.args['input_shape'] = left_input_batch.shape[2:4]
        left_features = self.pyramid_encoder(left_input_batch)
        right_features = self.pyramid_encoder(right_input_batch)

        left_0_sample_6 = left_features['f6']
        right_0_sample_6 = right_features['f6']

        dsi_6 = stereo_cost_volume_correlation(left_0_sample_6, right_0_sample_6, radius_x=self.args['radius_x'],
                                               stride=self.args['stride'])

        v6 = self.disparity_decoder_6(dsi_6)
        real_disp_v6 = self._make_disp(v6, self.scales[6])
        _disparity.append(real_disp_v6)

        u5 = F.interpolate(v6, size=(
            self.args['input_shape'][0] // self.scales[5], self.args['input_shape'][1] // self.scales[5]),
                           mode='bilinear', align_corners=False) * 20. / self.scales[5]

        if self.args['bulkhead']:
            # u5.detach().requires_grad = False
            u5 = Variable(u5, requires_grad=False)

        left_0_sample_5 = left_features['f5']

        if self.args['warping']:
            right_0_sample_5 = self._linear_warping(right_features['f5'],
                                                    self._build_indeces(torch.cat([u5, torch.zeros_like(u5)], dim=1)))

        else:
            right_0_sample_5 = right_features['f5']

        dsi_5 = stereo_cost_volume_correlation(left_0_sample_5, right_0_sample_5, self.args['radius_x'],
                                               self.args['stride'])
        volume = torch.cat([dsi_5, u5], dim=1)
        v5 = self.disparity_decoder_5(volume)
        real_disp_v5 = self._make_disp(v5, self.scales[5])
        _disparity.append(real_disp_v5)

        u4 = F.interpolate(v5, size=(
            self.args['input_shape'][0] // self.scales[4], self.args['input_shape'][1] // self.scales[4]),
                           mode='bilinear', align_corners=False) * 20. / self.scales[4]
        if self.args['bulkhead']:
            # u5.detach().requires_grad = False
            u4 = Variable(u4, requires_grad=False)

        left_0_sample_4 = left_features['f4']
        if self.args['warping']:
            right_0_sample_4 = self._linear_warping(right_features['f4'],
                                                    self._build_indeces(torch.cat([u4, torch.zeros_like(u4)], dim=1)))
        else:
            right_0_sample_4 = right_features['f4']

        dsi_4 = stereo_cost_volume_correlation(left_0_sample_4, right_0_sample_4, self.args['radius_x'],
                                               self.args['stride'])
        volume = torch.cat([dsi_4, u4], dim=1)
        v4 = self.disparity_decoder_4(volume)
        real_disp_v4 = self._make_disp(v4, self.scales[4])
        _disparity.append(real_disp_v4)
        u3 = F.interpolate(v4, size=(
            self.args['input_shape'][0] // self.scales[3], self.args['input_shape'][1] // self.scales[3]),
                           mode='bilinear', align_corners=False) * 20. / self.scales[3]
        if self.args['bulkhead']:
            u3 = Variable(u3, requires_grad=False)

        left_0_sample_3 = left_features['f3']
        if self.args['warping']:
            right_0_sample_3 = self._linear_warping(right_features['f3'],
                                                    self._build_indeces(torch.cat([u3, torch.zeros_like(u3)], dim=1)))
        else:
            right_0_sample_3 = right_features['f3']

        dsi_3 = stereo_cost_volume_correlation(left_0_sample_3, right_0_sample_3, self.args['radius_x'],
                                               self.args['stride'])
        volume = torch.cat([dsi_3, u3], dim=1)
        v3 = self.disparity_decoder_3(volume)
        real_disp_v3 = self._make_disp(v3, self.scales[3])
        _disparity.append(real_disp_v3)
        u2 = F.interpolate(v3, size=(
            self.args['input_shape'][0] // self.scales[2], self.args['input_shape'][1] // self.scales[2]),
                           mode='bilinear', align_corners=False) * 20. / self.scales[2]
        if self.args['bulkhead']:
            u2 = Variable(u2, requires_grad=False)

        left_0_sample_2 = left_features['f2']
        if self.args['warping']:

            right_0_sample_2 = self._linear_warping(right_features['f2'],
                                                    self._build_indeces(torch.cat([u2, torch.zeros_like(u2)], dim=1)))

        else:
            right_0_sample_2 = right_features['f2']

        dsi_2 = stereo_cost_volume_correlation(left_0_sample_2, right_0_sample_2, self.args['radius_x'],
                                               self.args['stride'])

        volume = torch.cat([dsi_2, u2], dim=1)
        v2_init = self.disparity_decoder_2(volume)

        if self.args['context_net']:
            volume = torch.cat([left_features['f2'], v2_init], dim=1)
            context_7 = self.refinement_module(volume)
            v2 = v2_init + context_7
            real_disp_v2_context = self._make_disp(v2, self.scales[2])
            _disparity.append(real_disp_v2_context)
        else:
            v2 = v2_init
            real_disp_v2 = self._make_disp(v2, self.scales[2])
            self._disparity.append(real_disp_v2)

        rescaled_prediction = F.interpolate(v2, size=self.args['input_shape'], mode='bilinear', align_corners=False)
        rescaled_prediction = self.relu(rescaled_prediction * -20.)
        restore_prediction = resize_image_with_crop_or_pad_2(rescaled_prediction, self.restore_shape[0],
                                                           self.restore_shape[1])
        _disparity.append(restore_prediction)
        return _disparity

    def _init_weights(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)

    def _linear_warping(self, imgs, coords):
        shape = coords.shape
        coord_b, coord_x, coord_y = torch.split(coords, [1, 1, 1], dim=1)

        coord_x = coord_x.float()
        coord_y = coord_y.float()

        x0 = torch.floor(coord_x)
        x1 = x0 + 1
        y0 = torch.floor(coord_y)

        y_max = (imgs.shape[2] - 1).__float__()
        x_max = (imgs.shape[3] - 1).__float__()
        zero = torch.zeros([1], dtype=torch.float32)

        x0_safe = torch.clamp(x0, zero[0], x_max)
        y0_safe = torch.clamp(y0, zero[0], y_max)
        x1_safe = torch.clamp(x1, zero[0], x_max)

        # bilinear interp weights, with points outside the grid having weight 0
        wt_x0 = (x1 - coord_x) * torch.eq(x0, x0_safe).float()
        wt_x1 = (coord_x - x0) * torch.eq(x1, x1_safe).float()

        im00 = gather_nd_torch(imgs, torch.cat([coord_b, y0_safe, x0_safe], dim=1)).float()
        im01 = gather_nd_torch(imgs, torch.cat([coord_b, y0_safe, x1_safe], dim=1)).float()

        # output = torch.add([wt_x0*im00,wt_x1*im01])
        output = (wt_x0 * im00).add(wt_x1 * im01)
        return output

    def _build_indeces(self, coords):

        device = coords.device
        batches = coords.shape[0]
        height = coords.shape[2]
        width = coords.shape[3]

        # pixel_coords = np.ones((1, 2, height, width))
        batches_coords = np.ones((batches, 1, height, width))

        for i in range(0, batches):
            batches_coords[i][:][:][:] = i

        # build pixel coordinates and their disparity
        # for i in range(0, height):
        #     for j in range(0, width):
        #         pixel_coords[0][0][i][j] = j
        #         pixel_coords[0][1][i][j] = i
        yv, xv = torch.meshgrid([torch.arange(height), torch.arange(width)])
        pixel_coords = torch.stack((xv, yv), 0).reshape(1, 2, height, width).type(torch.float32)

        # pixel_coords = torch.Tensor(pixel_coords, dtype=torch.float32)
        pixel_coords = torch.FloatTensor(pixel_coords)
        batches_coords = torch.FloatTensor(batches_coords)

        pixel_coords = pixel_coords.to(device)
        batches_coords = batches_coords.to(device)

        output = torch.cat([batches_coords, pixel_coords + coords], dim=1)
        return output

    def _make_disp(self, op, scale=None):
        # 源代码需要*-20,不理解。加relu可以认为是视差没有负值
        op = self.relu(op * -20)
        # op = self.relu(op)
        op = F.interpolate(op, size=self.restore_shape, mode='bilinear', align_corners=False)
        return op

    def _check_before_run(self, args):
        """
        Check that args contains everything that is needed
        Valid Keys for args:
            warping: boolean to enable or disable warping
            context_net: boolean to enable or disable context_net
            radius_d: kernel side used for computing correlation map
            stride: stride used to compute correlation map
        """
        portion_options = ['BEGIN', 'END']
        if 'warping' not in args:
            print('WARNING: warping flag not setted, setting default True value')
            args['warping'] = True
        if 'context_net' not in args:
            print('WARNING: context_net flag not setted, setting default True value')
            args['context_net'] = True
        if 'radius_x' not in args:
            print('WARNING: radius_d not setted, setting default value 2')
            args['radius_x'] = 2
        if 'stride' not in args:
            print('WARNING: stride not setted, setting default value 1')
            args['stride'] = 1
        if 'bulkhead' not in args:
            args['bulkhead'] = False
        if 'split_layers' not in args:
            print('WARNING: no split points selected, the network will flow without interruption')
            args['split_layers'] = [None]
        if 'train_portion' not in args:
            print('WARNING: train_portion not specified, using default END')
            args['train_portion'] = 'END' if args['split_layers'] != [None] else 'BEGIN'
        elif args['train_portion'] not in portion_options:
            raise Exception('Invalid portion options {}'.format(args['train_portion']))
        if 'sequence' not in args:
            print('WARNING: sequence flag not setted, configuring the network for single image adaptation')
            args['sequence'] = False
        if 'is_training' not in args:
            print('WARNING: flag for trainign not setted, using default False')
            args['is_training'] = False


def madnet(args):
    model = MadNet(pyramid_encoder=Pyramid_Encoder,
                   disparity_decoder=Disparity_Decoder,
                   refinement_module=Refinement_Module,
                   args=args)
    return model


if __name__ == '__main__':
    args = {}
    args['radius_x'] = 2
    args['stride'] = 1
    args['input_shape'] = (320, 1216)
    args['bulkhead'] = True
    args['warping'] = True
    args['context_net'] = True
    net = madnet(args=args)
    x = torch.randn(1, 3, 320, 1216)
    y = net(x, x)
    print(net)
    print(y)

    print(1)
