from pathlib import Path

import torch.nn as nn
import torch
import torchvision
from torchsummary import summary

from backbone.resnet import resnet50
from fpn_neck import FeaturePyramidNetwork as fpn
from fpn_neck import LastLevelP6P7
from head import ClsCntRegHead
from loss import GenTargets, Loss, coords_featureMap2original


class FCOS(nn.Module):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_"""

    def __init__(self, cfg='model.yaml'):
        super(FCOS, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict
        if self.yaml['backbone'] == 'resnet50':
            self.backbone = resnet50(pretrained=self.yaml['pretrained'], if_include_top=False)
        elif self.yaml['backbone'] == 'other':
            self.backbone = None
        else:
            raise NotImplementedError("backbone only implemented ['resnet50']")

        self.neck = fpn(in_channels_list=self.yaml['fpn_in_channel'],
                        out_channels=self.yaml['fpn_out_channel'],
                        extra_blocks=LastLevelP6P7(self.yaml['fpn_out_channel'], self.yaml['fpn_out_channel']) if
                        self.yaml['use_p5'] else None)

        self.head = ClsCntRegHead(in_channel=self.yaml['fpn_out_channel'],
                                  out_channel=self.yaml['head_out_channle'],
                                  class_num=self.yaml['class_num'],
                                  GN=self.yaml['use_GN_head'],
                                  cnt_on_reg=self.yaml['add_centerness'],
                                  prior=self.yaml['prior'])

    def forward(self, x):
        """

        :param x: tensor [batch_size, 3, h, w] / imgs
        :return:         list [cls_logits,cnt_logits,reg_preds]
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        """
        backbone_outputs = self.backbone(x)
        fpn_outputs = self.neck(backbone_outputs)
        outputs = self.head(fpn_outputs)
        if self.training:
            if self.yaml['freeze_bn']:
                def freeze_bn(module):
                    if isinstance(module, nn.BatchNorm2d):
                        module.eval()
                    classname = module.__class__.__name__
                    if classname.find('BatchNorm') != -1:
                        for p in module.parameters(): p.requires_grad = False

                self.apply(freeze_bn)
            if self.yaml['freeze_stage_1']:
                self.backbone.freeze_stages(1)
        return outputs


class ClipBoxes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_imgs, batch_boxes):
        batch_boxes = batch_boxes.clamp_(min=0)
        h, w = batch_imgs.shape[2:]
        batch_boxes[..., [0, 2]] = batch_boxes[..., [0, 2]].clamp_(max=w - 1)
        batch_boxes[..., [1, 3]] = batch_boxes[..., [1, 3]].clamp_(max=h - 1)
        return batch_boxes


class FCOSDetector(nn.Module):
    def __init__(self, mode='training', cfg='model.yaml'):
        super(FCOSDetector, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

        self.mode = mode
        self.body = FCOS(cfg)
        if mode == 'training':
            self.target_layer = GenTargets(self.yaml['strides'], self.yaml['limit_range'])
            self.loss_layer = Loss(cfg='models/model.yaml')
        elif mode == 'inference':
            self.detection_head = DetectHead(
                score_threshold=self.yaml['score_threshold'],
                nms_iou_threshold=self.yaml['nms_iou_threshold'],
                max_detection_boxes_num=self.yaml['max_detection_boxes_num'],
                strides=self.yaml['strides'],
                cfg='models/model.yaml'
            )
            self.clip_boxes = ClipBoxes()

    def forward(self, inputs):
        """

        :param inputs:
                [training] list  [batch_imgs,batch_boxes,batch_classes]
                [inference] img
        :return:
        """
        if self.mode == 'training':
            batch_imgs, batch_boxes, batch_classes = inputs
            # 网络输出类别，cnt置信度，以及四个预测的偏移量.ltrb.
            out = self.body(batch_imgs)
            # 对类别以及box进行编码。将gt的类别，box转为与网络输出对应的形式
            # box:[xmin,ymin,xmax,ymax] -> [l,t,r,b]:即box边界距离网络中心点的偏移量
            targets = self.target_layer([out, batch_boxes, batch_classes])
            # 计算损失
            losses = self.loss_layer([out, targets])
            return losses
        elif self.mode == 'inference':
            # raise NotImplementedError("no implement inference model")
            '''
            for inference mode, img should preprocessed before feeding in net 
            '''
            batch_imgs = inputs
            out = self.body(batch_imgs)
            scores, classes, boxes = self.detection_head(out)
            boxes = self.clip_boxes(batch_imgs, boxes)
            return scores, classes, boxes


class DetectHead(nn.Module):
    def __init__(self, score_threshold, nms_iou_threshold, max_detection_boxes_num, strides, cfg='model.yaml'):
        super(DetectHead, self).__init__()
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detection_boxes_num = max_detection_boxes_num
        self.strides = strides
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict

    def forward(self, inputs):
        """

        :param inputs: list [cls_logits,cnt_logits,reg_preds]
                cls_logits  list contains five [batch_size,class_num,h,w]
                cnt_logits  list contains five [batch_size,1,h,w]
                reg_preds   list contains five [batch_size,4,h,w]  注意：网络预测的是目标到网格中心点的偏移量，即ltrb。
        :return:
        """
        cls_logits, coords = self._reshape_cat_out(inputs[0], self.strides)  # [batch_size,sum(_h*_w),class_num]
        cnt_logits, _ = self._reshape_cat_out(inputs[1], self.strides)  # [batch_size,sum(_h*_w),1]
        reg_preds, _ = self._reshape_cat_out(inputs[2], self.strides)  # [batch_size,sum(_h*_w),4]

        cls_preds = cls_logits.sigmoid_()  # 归一化
        cnt_preds = cnt_logits.sigmoid_()

        coords = coords.cuda() if torch.cuda.is_available() else coords

        # 找出每个位置 类别最大的置信度，返回分数以及索引
        cls_scores, cls_classes = torch.max(cls_preds, dim=-1)  # [batch_size,sum(_h*_w)]

        if self.yaml['add_centerness']:
            # 根据论文，最终的置信度等于根号下类别置信度*cnt分支预测的值
            cls_scores = torch.sqrt(cls_scores * (cnt_preds.squeeze(dim=-1)))  # [batch_size,sum(_h*_w)]
        # 因为类别列表是从1开始的，0为背景类。而索引是从0开始的，所以要+1.
        cls_classes = cls_classes + 1  # [batch_size,sum(_h*_w)]

        # 对网络预测的偏移量，进行解码，即将ltrb->box。coords：根据步长产生的网络中心点坐标，reg_preds:网络预测的偏移量。boxes：[xmin,ymin,xmax,ymax]
        boxes = self._coords2boxes(coords, reg_preds)  # [batch_size,sum(_h*_w),4]

        # 选择top k的预测框
        max_num = min(self.max_detection_boxes_num,
                      cls_scores.shape[-1])  # cls_scores：[batch_size,sum(_h*_w)] 5层特征图size之和
        # 根据类别置信度，返回前top num个框，降序排列，返回对应的索引
        topk_ind = torch.topk(cls_scores, max_num, dim=-1, largest=True, sorted=True)[1]  # [batch_size,max_num]
        _cls_scores = []
        _cls_classes = []
        _boxes = []
        for batch in range(cls_scores.shape[0]):  # cls_scores：[batch_size,sum(_h*_w)] 5层特征图size之和
            _cls_scores.append(cls_scores[batch][topk_ind[batch]])  # [max_num]， 前max_nums置信度分数
            _cls_classes.append(cls_classes[batch][topk_ind[batch]])  # [max_num]， 前max_nums对应的类别
            _boxes.append(boxes[batch][topk_ind[batch]])  # [max_num,4]， 前max_nums的box
        cls_scores_topk = torch.stack(_cls_scores, dim=0)  # [batch_size,max_num]
        cls_classes_topk = torch.stack(_cls_classes, dim=0)  # [batch_size,max_num]
        boxes_topk = torch.stack(_boxes, dim=0)  # [batch_size,max_num,4]
        assert boxes_topk.shape[-1] == 4
        return self._post_process([cls_scores_topk, cls_classes_topk, boxes_topk])

    def _post_process(self, preds_topk):
        """

        :param preds_topk: list contains three elem [cls_scores_topk,cls_classes_topk,boxes_topk]
            cls_scores_topk [batch_size,max_num]
            cls_classes_topk [batch_size,max_num]
            boxes_topk [batch_size,max_num,4]
        :return:
        """
        _cls_scores_post = []
        _cls_classes_post = []
        _boxes_post = []
        cls_scores_topk, cls_classes_topk, boxes_topk = preds_topk
        for batch in range(cls_classes_topk.shape[0]):
            # 选择置信度大于阈值的位置
            mask = cls_scores_topk[batch] >= self.score_threshold
            _cls_scores_b = cls_scores_topk[batch][mask]  # [N]
            _cls_classes_b = cls_classes_topk[batch][mask]  # [N]
            _boxes_b = boxes_topk[batch][mask]  # [N,4]

            # 根据每个类别进行过滤，只对同一种类别进行计算IOU和阈值
            nms_ind = self.batched_nms(_boxes_b, _cls_scores_b, _cls_classes_b, self.nms_iou_threshold)

            # 等价于 pytorch官网实现的batch_nms
            # nms_ind = torchvision.ops.batched_nms(_boxes_b, _cls_scores_b, _cls_classes_b, self.nms_iou_threshold)

            _cls_scores_post.append(_cls_scores_b[nms_ind])
            _cls_classes_post.append(_cls_classes_b[nms_ind])
            _boxes_post.append(_boxes_b[nms_ind])

        scores, classes, boxes = torch.stack(_cls_scores_post, dim=0), torch.stack(_cls_classes_post,
                                                                                   dim=0), torch.stack(_boxes_post,
                                                                                                       dim=0)

        return scores, classes, boxes

    def _reshape_cat_out(self, inputs, strides):
        """

        :param inputs: list contains five [batch_size,c,_h,_w]
        :param strides: [8, 16, 32, 64, 128]
        :return:
            out [batch_size,sum(_h*_w),c]
            coords [sum(_h*_w),2]
        """
        batch_size = inputs[0].shape[0]
        c = inputs[0].shape[1]
        out = []
        coords = []
        for pred, stride in zip(inputs, strides):
            pred = pred.permute(0, 2, 3, 1)  # [batch_size,c,_h,_w] -> [batch_size,_h,_w,c]
            coord = coords_featureMap2original(pred, stride).to(device=pred.device)  # [n,2] n=_h*_w
            pred = torch.reshape(pred, [batch_size, -1, c])  # [batch_size,_h,_w,c] -> [batch_size,_h*_w,c]
            out.append(pred)
            coords.append(coord)
        return torch.cat(out, dim=1), torch.cat(coords, dim=0)

    def _coords2boxes(self, coords, offsets):
        """

        :param coords: [sum(_h*_w),2]
        :param offsets: [batch_size,sum(_h*_w),4] ltrb
        :return: [batch_size,sum(_h*_w),4] 4:box:[xmin,ymin,xmax,ymax]
        """
        x1y1 = coords[None, :, :] - offsets[..., :2]
        x2y2 = coords[None, :, :] + offsets[..., 2:]  # [batch_size,sum(_h*_w),2]
        boxes = torch.cat([x1y1, x2y2], dim=-1)  # [batch_size,sum(_h*_w),4]
        return boxes

    @staticmethod
    def box_nms(boxes, scores, thr):
        """

        :param boxes: [N,4]
        :param scores: [N]
        :param thr: float
        :return:
        """
        if boxes.shape[0] == 0:
            return torch.zeros(0, device=boxes.device).long()
        assert boxes.shape[-1] == 4
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()
                keep.append(i)

            xmin = x1[order[1:]].clamp(min=float(x1[i]))
            ymin = y1[order[1:]].clamp(min=float(y1[i]))
            xmax = x2[order[1:]].clamp(max=float(x2[i]))
            ymax = y2[order[1:]].clamp(max=float(y2[i]))
            inter = (xmax - xmin).clamp(min=0) * (ymax - ymin).clamp(min=0)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            idx = (iou <= thr).nonzero().squeeze()
            if idx.numel() == 0:
                break
            order = order[idx + 1]
        return torch.LongTensor(keep)

    def batched_nms(self, boxes, scores, idxs, iou_threshold):
        """

        :param boxes: [N,4]
        :param scores: [N]
        :param idxs: [N]
        :param iou_threshold:
        :return:
        """
        # numel():求和
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        # strategy: in order to perform NMS independently per class.
        # we add an offset to all the boxes. The offset is dependent
        # only on the class idx, and is large enough so that boxes
        # from different classes do not overlap
        max_coordinate = boxes.max()
        # .to():改变加载数据的设备，cpu or gpu
        offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]
        keep = self.box_nms(boxes_for_nms, scores, iou_threshold)
        return keep


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    body = FCOS().to(device)
    print(body)
    y = body(torch.randn(2, 3, 224, 224).to(device))
    print(y)

    model = FCOSDetector(mode='inference').to(device)
    model.eval()
    y = model.forward(torch.randn(2,3,224,224).to(device))
    # summary(model, input_size=(3, 224, 224))
    print(y)
