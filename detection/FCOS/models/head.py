import torch.nn as nn
import torch
import math


class ScaleExp(nn.Module):
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.tensor([init_value], dtype=torch.float32))

    def forward(self, x):
        return torch.exp(x * self.scale)


class ClsCntRegHead(nn.Module):
    def __init__(self, in_channel, out_channel, class_num, GN=True, cnt_on_reg=True, prior=0.01):
        super(ClsCntRegHead, self).__init__()
        self.cnt_on_reg = cnt_on_reg

        cls_branch = []
        reg_branch = []

        conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True)
        gn = nn.GroupNorm(32, out_channel)
        for i in range(4):
            cls_branch.append(conv)
            if GN:
                cls_branch.append(gn)
            cls_branch.append(nn.ReLU(inplace=True))

            reg_branch.append(conv)
            if GN:
                reg_branch.append(gn)
            reg_branch.append(nn.ReLU(inplace=True))

        self.cls = nn.Sequential(*cls_branch)
        self.reg = nn.Sequential(*reg_branch)

        self.cls_logits = nn.Conv2d(out_channel, class_num, kernel_size=3, stride=1, padding=1)
        self.cnt_logits = nn.Conv2d(out_channel, 1, kernel_size=3, stride=1, padding=1)
        self.reg_pred = nn.Conv2d(out_channel, 4, kernel_size=3, stride=1, padding=1)

        self.apply(self.init_conv_RandomNormal)
        nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior) / prior))
        self.scale_exp = nn.ModuleList([ScaleExp(1.0) for _ in range(5)])

    def init_conv_RandomNormal(self, module, std=0.01):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=std)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, inputs):
        """

        :param inputs: [p3-p7]
        :return:
        """
        cls_logits = []
        cnt_logits = []
        reg_preds = []
        for index, P in enumerate(inputs):
            cls_conv_out = self.cls(inputs[P])
            reg_conv_out = self.reg(inputs[P])

            cls_logits.append(self.cls_logits(cls_conv_out))
            if not self.cnt_on_reg:
                cnt_logits.append(self.cnt_logits(cls_conv_out))
            else:
                cnt_logits.append(self.cnt_logits(reg_conv_out))
            reg_preds.append(self.scale_exp[index](self.reg_pred(reg_conv_out)))
        return cls_logits, cnt_logits, reg_preds
