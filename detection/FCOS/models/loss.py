from pathlib import Path

import torch
import torch.nn as nn


def coords_featureMap2original(feature, stride):
    """
    transfor one feature map coords to orig img coords
    :param feature: [batch_size,h,w,c]
    :param stride: int
    :return: [n,2]
    """
    h, w = feature.shape[1:3]
    shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)  # w
    shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)  # h
    shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x)  # [h, h], [w, w]
    shifts_x = torch.reshape(shifts_x, [-1])
    shifts_y = torch.reshape(shifts_y, [-1])
    coords = torch.stack([shifts_x, shifts_y], -1) + stride // 2  # [n, 2] n = w*h
    return coords


# coords_featureMap2original(torch.randn(1, 14, 14, 1), 8)


class GenTargets(nn.Module):
    def __init__(self, strides: list, limit_range: list):
        super(GenTargets, self).__init__()
        self.strides = strides
        self.limit_range = limit_range
        assert len(strides) == len(limit_range)

    def forward(self, inputs):
        """

        :param inputs:
        [0]list [cls_logits,cnt_logits,reg_preds]
            cls_logits  list contains five [batch_size,class_num,h,w]
            cnt_logits  list contains five [batch_size,1,h,w]
            reg_preds   list contains five [batch_size,4,h,w]
        [1]gt_boxes [batch_size,m,4]  FloatTensor
        [2]classes [batch_size,m]  LongTensor
        :return:
        cls_targets:[batch_size,sum(_h*_w),1]
        cnt_targets:[batch_size,sum(_h*_w),1]
        reg_targets:[batch_size,sum(_h*_w),4]
        """
        cls_logits, cnt_logits, reg_preds = inputs[0]
        gt_boxes = inputs[1]
        classes = inputs[2]
        cls_targets_all_level = []
        cnt_targets_all_level = []
        reg_targets_all_level = []
        assert len(self.strides) == len(cls_logits)
        for level in range(len(cls_logits)):
            level_out = [cls_logits[level], cnt_logits[level], reg_preds[level]]
            level_targets = self._gen_level_targets(level_out, gt_boxes, classes, self.strides[level],
                                                    self.limit_range[level])
            cls_targets_all_level.append(level_targets[0])
            cnt_targets_all_level.append(level_targets[1])
            reg_targets_all_level.append(level_targets[2])
        return torch.cat(cls_targets_all_level, dim=1), torch.cat(cnt_targets_all_level, dim=1), torch.cat(
            reg_targets_all_level, dim=1)

    def _gen_level_targets(self, out, gt_boxes, classes, stride, limit_range, sample_radiu_ratio=1.5):
        """

        :param out: [[batch_size,class_num,h,w],[batch_size,1,h,w],[batch_size,4,h,w]]
        :param gt_boxes: [batch_size,m,4]  4:[xmin,ymin,xmax,ymax]
        :param classes: [batch_size,m]
        :param stride: int
        :param limit_range: list [min,max]
        :param sample_radiu_ratio: float
        :return:  cls_targets,cnt_targets,reg_targets
        """
        cls_logits, cnt_logits, reg_preds = out
        batch_size = cls_logits.shape[0]
        class_num = cls_logits.shape[1]
        m = gt_boxes.shape[1]

        cls_logits = cls_logits.permute(0, 2, 3, 1)  # [batch_size,class_num,h,w] -> [batch_size,h,w,class_num]
        cnt_logits = cnt_logits.permute(0, 2, 3, 1)  # [batch_size,1,h,w] -> [batch_size,h,w,1]
        reg_preds = reg_preds.permute(0, 2, 3, 1)  # [batch_size,4,h,w] ->[batch_size,h,w,4]

        coords = coords_featureMap2original(cls_logits, stride).to(device=gt_boxes.device)  # [h*w,2]

        cls_logits = cls_logits.reshape((batch_size, -1, class_num))  # [batch_size, h*w, class_num]
        # cnt_logits = cnt_logits.reshape((batch_size, -1, 1))  # [batch_size, h*w, 1]
        # reg_preds = reg_preds.shape((batch_size, -1, 4))  # [batch_size, h*w, 4]

        h_mul_w = cls_logits.shape[1]

        x = coords[:, 0]  # h*w
        y = coords[:, 1]  # h*w
        """
        x[None, :, None]: [1,h*w,1]
        gt_boxes[..., 0]: [batch_size, m]
        gt_boxes[..., 0][:,None,:]: [batch_size, 1, m]
        """
        # 可以理解为bs张图片，每张图片m个目标，即计算h*w大小的特征图上每个像素点映射回原图上距离gt框的偏移量
        l_off = x[None, :, None] - gt_boxes[..., 0][:, None, :]  # 左 [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        t_off = y[None, :, None] - gt_boxes[..., 1][:, None, :]  # 上 [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        r_off = gt_boxes[..., 2][:, None, :] - x[None, :, None]  # 右 [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        b_off = gt_boxes[..., 3][:, None, :] - y[None, :, None]  # 下 [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        ltrb_off = torch.stack([l_off, t_off, r_off, b_off], dim=-1)  # [batch_size,h*w,m,4]

        # 取每个位置距离GT框的最小偏移量，[0]:具体的值 [1]:最小值的位置索引。即在ltrb，这四个偏移量中取最小值。
        off_min = torch.min(ltrb_off, dim=-1)[0]  # [batch_size, h*w, m]
        # 取每个位置距离GT框的最大偏移量，[0]:具体的值 [1]:最小值的位置索引。即在ltrb，这四个偏移量中取最大值
        off_max = torch.max(ltrb_off, dim=-1)[0]  # [batch_size, h*w,m]

        """
        yolox
        找正样本的两种方式：
            1. 寻找那些在gt框内的点
            2. 寻找那些在 以gt框中心点为中心，半径为stride * sample_radiu_ratio为半径的正方形内的点
        fcos
        找正样本的方式
            同时满足上述两点
        """

        # 判断每个位置是否在GT框内
        mask_in_gtboxes = off_min > 0  # [batch_size, h*w, m]
        # 类似于yolo,不同大小特征层分配不同大小的anchor。这里是不同大小的特征层分配一定范围内的偏移量
        mask_in_level = (off_max > limit_range[0]) & (off_max < limit_range[1])  # [batch_size, h*w, m]

        # 半径，1.5倍的stride长度
        radiu = stride * sample_radiu_ratio

        # gt框中心点 x
        gt_center_x = (gt_boxes[..., 0] + gt_boxes[..., 2]) / 2  # ceter = (min+max)/2
        # gt框中心点 y
        gt_center_y = (gt_boxes[..., 1] + gt_boxes[..., 3]) / 2  # [batch_size, h*w,m]

        # 计算特征图上每个像素点投射回原图距离gt框中心点偏移量
        c_l_off = x[None, :, None] - gt_center_x[:, None, :]  # [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        c_t_off = y[None, :, None] - gt_center_y[:, None, :]
        c_r_off = gt_center_x[:, None, :] - x[None, :, None]
        c_b_off = gt_center_y[:, None, :] - y[None, :, None]
        c_ltrb_off = torch.stack([c_l_off, c_t_off, c_r_off, c_b_off], dim=-1)  # [batch_size,h*w,m,4]
        c_off_max = torch.max(c_ltrb_off, dim=-1)[0]  # [batch_size, h*w, m]
        mask_center = c_off_max < radiu

        # 得到正样本
        mask_pos = mask_in_gtboxes & mask_in_level & mask_center  # [batch_size,h*w,m]

        # 计算每个位置，box的面积
        areas = (ltrb_off[..., 0] + ltrb_off[..., 2]) * (ltrb_off[..., 1] + ltrb_off[..., 3])  # [batch_size,h*w,m]
        # ~:取反
        areas[~mask_pos] = 999999999
        # 取得每个位置，bbox面积最小的索引
        # 根据论文介绍，如果同一特征层中，某个位置同时位于多个bbox框内，则选择面积最小的那个bbox作为该位置的回归对象
        areas_min_index = torch.min(areas, dim=-1)[1]  # #[batch_size,h*w]

        # 该批次中该特征图每个位置上的回归目标
        reg_targets = ltrb_off[torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_index.unsqueeze(dim=-1),
                                                                                  1)]  # [batch_size*h*w,4]
        reg_targets = torch.reshape(reg_targets, (batch_size, -1, 4))  # [batch_size,h*w,4]

        # 分类
        # torch.broadcast_tensors()是一个将tensor扩充的函数,广播机制
        # 返回值有两个结果[0]或者[1]。两者形状相同，但[1]填充的值是0
        # classes[:, None, :]: [batch_size,m] -> [batch_size,1,m]
        # areas: [batch_size,h*w,m]
        classes = torch.broadcast_tensors(classes[:, None, :], areas.long())[0]  # [batch_size,h*w,m]
        cls_targets = classes[torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_index.unsqueeze(dim=-1),
                                                                                 1)]  # [batch_size*h*w,1]
        cls_targets = torch.reshape(cls_targets, (batch_size, -1, 1))  # [batch_size,h*w,1]

        # 获取上下左右偏移量的最值
        left_right_min = torch.min(reg_targets[..., 0], reg_targets[..., 2])  # [batch_size,h*w]
        left_right_max = torch.max(reg_targets[..., 0], reg_targets[..., 2])
        top_bottom_min = torch.min(reg_targets[..., 1], reg_targets[..., 3])
        top_bottom_max = torch.max(reg_targets[..., 1], reg_targets[..., 3])

        """
        由于远离目标中心位置会产生很多低质量的bbox。所以添加center-ness。sqrt是为了缓解ceter-ness的衰减。
        center-ness(可以理解为一种具有度量作用的概念，在这里称之为"中心度")，中心度取值为0,1之间，使用交叉熵损失进行训练。
        并把损失加入前面提到的损失函数中。测试时，将预测的中心度与相应的分类分数相乘，计算最终得分(用于对检测到的边界框进行排序)。
        因此，中心度可以降低远离对象中心的边界框的权重。因此，这些低质量边界框很可能被最终的非最大抑制（NMS）过程滤除，从而显着提高了检测性能
        """
        # 根据论文中的公式3，得到cnt_targets.
        cnt_targets = ((left_right_min * top_bottom_min) / (left_right_max * top_bottom_max + 1e-10)).sqrt().unsqueeze(
            dim=-1)  # [batch_size,h*w,1]

        assert reg_targets.shape == (batch_size, h_mul_w, 4)
        assert cls_targets.shape == (batch_size, h_mul_w, 1)
        assert cnt_targets.shape == (batch_size, h_mul_w, 1)

        mask_pos_2 = mask_pos.long().sum(dim=-1)  # [batch_size,h*w,m] -> [batch_size,h*w]
        # num_pos=mask_pos_2.sum(dim=-1)
        # assert num_pos.shape==(batch_size,)
        mask_pos_2 = mask_pos_2 >= 1  # 确定那些位置上有目标
        assert mask_pos_2.shape == (batch_size, h_mul_w)
        # 没有目标的位置，即背景，标签赋值为0
        cls_targets[~mask_pos_2] = 0  # [batch_size,h*w,1]
        # 没有目标的位置。。。。赋值为-1
        cnt_targets[~mask_pos_2] = -1
        # 没有目标的位置。。。。赋值为-1
        reg_targets[~mask_pos_2] = -1

        return cls_targets, cnt_targets, reg_targets


# x = torch.randn(1, 4, 2, 4)
# areas = torch.randn(1, 4, 2)
# areas_min_index = torch.min(areas, dim=-1)[1]
# off_min = torch.min(x, dim=-1)[0]
# mask_in_gtboxes = off_min > 0
# mask_pos_2 = mask_in_gtboxes.long().sum(dim=-1)
# p = [~mask_in_gtboxes]
# reg_targets = x[torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_index.unsqueeze(dim=-1), 1)]
# print(x)

class Loss(nn.Module):
    def __init__(self, cfg='model.yaml'):
        super(Loss, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

    def forward(self, inputs):
        """

        :param inputs: list [preds,targets]
                [0]preds:  list contains three elements [cls_logits,cnt_logits,reg_preds]
                [1]targets : list contains three elements [[batch_size,sum(_h*_w),1],[batch_size,sum(_h*_w),1],[batch_size,sum(_h*_w),4]]
        :return:
                loss
        """
        preds, targets = inputs
        cls_logits, cnt_logits, reg_preds = preds
        cls_targets, cnt_targets, reg_targets = targets
        mask_pos = (cnt_targets > -1).squeeze(dim=-1)  # [batch_size,sum(_h*_w)]
        cls_loss = self._compute_cls_loss(cls_logits, cls_targets, mask_pos).mean()  # []
        cnt_loss = self._compute_cnt_loss(cnt_logits, cnt_targets, mask_pos).mean()
        reg_loss = self._compute_reg_loss(reg_preds, reg_targets, mask_pos).mean()
        if self.yaml['add_centerness']:
            total_loss = cls_loss + cnt_loss + reg_loss
            return cls_loss, cnt_loss, reg_loss, total_loss
        else:
            total_loss = cls_loss + reg_loss + cnt_loss * 0.0
            return cls_loss, cnt_loss, reg_loss, total_loss

    def _compute_cls_loss(self, preds, targets, mask):
        """

        :param preds: list contains five level pred [batch_size,class_num,_h,_w]
        :param targets: [batch_size,sum(_h*_w),1]
        :param mask: [batch_size,sum(_h*_w)]
        :return:
        """
        batch_size = targets.shape[0]
        preds_reshape = []
        class_num = preds[0].shape[1]
        mask = mask.unsqueeze(dim=-1)  # [batch_size,sum(_h*_w),1]
        # mask=targets>-1#[batch_size,sum(_h*_w),1]
        num_pos = torch.sum(mask, dim=[1, 2]).clamp_(min=1).float()  # 正样本数量
        for pred in preds:
            pred = pred.permute(0, 2, 3, 1)  # [batch_size,_h,_w,class_num]
            pred = torch.reshape(pred, [batch_size, -1, class_num])  # [batch_size,_h*_w,class_num]
            preds_reshape.append(pred)
        preds = torch.cat(preds_reshape, dim=1)  # [batch_size,sum(_h*_w),class_num]
        assert preds.shape[:2] == targets.shape[:2]
        loss = []
        for batch_index in range(batch_size):
            pred_pos = preds[batch_index]  # [sum(_h*_w),class_num]
            target_pos = targets[batch_index]  # [sum(_h*_w),1]
            target_pos = (torch.arange(1, class_num + 1, device=target_pos.device)[None,
                          :] == target_pos).float()  # sparse-->onehot bool->float -> onehot
            loss.append(self._focal_loss_from_logits(pred_pos, target_pos).view(1))
        return torch.cat(loss, dim=0) / num_pos  # [batch_size,]

    def _compute_cnt_loss(self, preds, targets, mask):
        """

        :param preds: list contains five level pred [batch_size,1,_h,_w]
        :param targets: [batch_size,sum(_h*_w),1]
        :param mask: [batch_size,sum(_h*_w)]
        :return:
        """
        batch_size = targets.shape[0]
        c = targets.shape[-1]
        preds_reshape = []
        mask = mask.unsqueeze(dim=-1)
        # mask=targets>-1#[batch_size,sum(_h*_w),1]
        num_pos = torch.sum(mask, dim=[1, 2]).clamp_(min=1).float()  # 正样本位置数量
        for pred in preds:
            pred = pred.permute(0, 2, 3, 1)
            pred = torch.reshape(pred, [batch_size, -1, c])
            preds_reshape.append(pred)
        preds = torch.cat(preds_reshape, dim=1)
        assert preds.shape == targets.shape  # [batch_size,sum(_h*_w),1]
        loss = []
        for batch_index in range(batch_size):
            pred_pos = preds[batch_index][mask[batch_index]]  # [num_pos_b,]
            target_pos = targets[batch_index][mask[batch_index]]  # [num_pos_b,]
            assert len(pred_pos.shape) == 1
            loss.append(
                nn.functional.binary_cross_entropy_with_logits(input=pred_pos, target=target_pos, reduction='sum').view(
                    1))
            # loss.append(torch.nn.BCEWithLogitsLoss(pred_pos, target_pos, reduction='sum').view(1))
            # torch.nn.BCEWithLogitsLoss 和 torch.nn.functional.binary_cross_entropy_with_logits 实现相同的功能，且参数一致。其代码本质就是后者。
        return torch.cat(loss, dim=0) / num_pos  # [batch_size,]

    def _compute_reg_loss(self, preds, targets, mask, mode='giou'):
        """

        :param preds: list contains five level pred [batch_size,4,_h,_w]
        :param targets: [batch_size,sum(_h*_w),4]
        :param mask: [batch_size,sum(_h*_w)]
        :param mode: iou loss type
        :return:
        """
        batch_size = targets.shape[0]
        c = targets.shape[-1]
        preds_reshape = []
        # mask=targets>-1#[batch_size,sum(_h*_w),4]
        num_pos = torch.sum(mask, dim=1).clamp_(min=1).float()  # 正样本匹配位置数量
        for pred in preds:
            pred = pred.permute(0, 2, 3, 1)
            pred = torch.reshape(pred, [batch_size, -1, c])
            preds_reshape.append(pred)
        preds = torch.cat(preds_reshape, dim=1)
        assert preds.shape == targets.shape  # [batch_size,sum(_h*_w),4]
        loss = []
        for batch_index in range(batch_size):
            pred_pos = preds[batch_index][mask[batch_index]]  # [num_pos_b,4]
            target_pos = targets[batch_index][mask[batch_index]]  # [num_pos_b,4]
            assert len(pred_pos.shape) == 2
            if mode == 'iou':
                loss.append(self._iou_loss(pred_pos, target_pos).view(1))
            elif mode == 'giou':
                loss.append(self._giou_loss(pred_pos, target_pos).view(1))
            else:
                raise NotImplementedError("reg loss only implemented ['iou','giou']")
        return torch.cat(loss, dim=0) / num_pos  # [batch_size,]

    def _focal_loss_from_logits(self, preds, targets, gamma=2.0, alpha=0.25, reduction='sum'):
        """

        :param preds: [n,class_num]
        :param targets: [n,class_num]
        :param gamma:  float
        :param alpha: float
        :return: loss
        """
        preds = torch.sigmoid(preds)
        p_t = targets * preds + (1 - targets) * (1 - preds)
        alpha_factor = targets * alpha + (1 - targets) * (1 - alpha)
        modulating_factor = (1.0 - p_t) ** gamma
        loss = -alpha_factor * modulating_factor * p_t.log()
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def _iou_loss(self, preds, targets, reduction='sum'):
        """

        :param preds: [n,4] ltrb 中心点到上下左右的偏移量
        :param targets: [n,4]       中心点到上下左右的偏移量
        :param reduction:
        :return:
        """
        lt = torch.min(preds[:, :2], targets[:, :2])  # 左上偏移量的最小值
        rb = torch.min(preds[:, 2:], targets[:, 2:])  # 右下偏移量的最小值
        wh = (rb + lt).clamp(min=0)  # w,h = 偏移量值相加
        overlap = wh[:, 0] * wh[:, 1]  # [n]
        area1 = (preds[:, 2] + preds[:, 0]) * (preds[:, 3] + preds[:, 1])
        area2 = (targets[:, 2] + targets[:, 0]) * (targets[:, 3] + targets[:, 1])
        iou = overlap / (area1 + area2 - overlap)
        loss = -iou.clamp(min=1e-6).log()
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def _giou_loss(self, preds, targets, reduction='sum'):
        """

        :param preds: [n,4] ltrb
        :param targets: [n,4]
        :return:
        """
        lt_min = torch.min(preds[:, :2], targets[:, :2])
        rb_min = torch.min(preds[:, 2:], targets[:, 2:])
        wh_min = (rb_min + lt_min).clamp(min=0)
        overlap = wh_min[:, 0] * wh_min[:, 1]  # [n]
        area1 = (preds[:, 2] + preds[:, 0]) * (preds[:, 3] + preds[:, 1])
        area2 = (targets[:, 2] + targets[:, 0]) * (targets[:, 3] + targets[:, 1])
        union = (area1 + area2 - overlap)
        iou = overlap / union

        lt_max = torch.max(preds[:, :2], targets[:, :2])
        rb_max = torch.max(preds[:, 2:], targets[:, 2:])
        wh_max = (rb_max + lt_max).clamp(0)
        G_area = wh_max[:, 0] * wh_max[:, 1]  # [n]

        giou = iou - (G_area - union) / G_area.clamp(1e-10)
        loss = 1. - giou
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss


# preds = [torch.ones([2, 3, 4, 4])] * 5
# targets = torch.ones([2, 4 * 4 * 5, 1])
# mask = torch.ones([2, 4 * 4 * 5], dtype=torch.bool)
# l = Loss()
# ll = l._compute_cls_loss(preds, targets, mask)
# print(ll)
