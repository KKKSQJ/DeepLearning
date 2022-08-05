import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.backbone import resnet


class SSPNet(nn.Module):
    def __init__(self, backbone, refine=False):
        super(SSPNet, self).__init__()
        backbone = resnet.__dict__[backbone](pretrained=True)
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2, self.layer3 = backbone.layer1, backbone.layer2, backbone.layer3
        self.refine = refine

    # img_s_list: list [[bs,3,h,w],[],[],[]], len()=shot
    # mask_s_list: list [[bs,h,w],[],[],[]]
    def forward(self, img_s_list, mask_s_list, img_q, mask_q):
        h, w = img_q.size()[-2:]

        # feature map of support images
        feature_s_list = []
        for k in range(len(img_s_list)):
            with torch.no_grad():
                s0 = self.layer0(img_s_list[k])
                s0 = self.layer1(s0)
            s0 = self.layer2(s0)
            s0 = self.layer3(s0)
            feature_s_list.append(s0)
            del s0

        # feature map of query image
        with torch.no_grad():
            q0 = self.layer0(img_q)
            q0 = self.layer1(q0)
        q0 = self.layer2(q0)
        # b c h w
        feature_q = self.layer3(q0)

        # foreground(target class) and background prototypes pooled from k support features
        feature_fg_list = []
        feature_bg_list = []
        supp_out_list = []

        for k in range(len(img_s_list)):
            # bs c -> 1 bs c
            feature_fg = self.masked_average_pooling(feature_s_list[k], (mask_s_list[k] == 1).float())[None, :]
            feature_bg = self.masked_average_pooling(feature_s_list[k], (mask_s_list[k] == 0).float())[None, :]

            feature_fg_list.append(feature_fg)
            feature_bg_list.append(feature_bg)

            if self.training:
                # bs h w
                supp_similarity_fg = F.cosine_similarity(feature_s_list[k], feature_fg.squeeze(0)[..., None, None],
                                                         dim=1)
                supp_similarity_bg = F.cosine_similarity(feature_s_list[k], feature_bg.squeeze(0)[..., None, None],
                                                         dim=1)
                # bs 2 h w
                supp_out = torch.cat((supp_similarity_bg[:, None, ...], supp_similarity_fg[:, None, ...]), dim=1) * 10.0

                supp_out = F.interpolate(supp_out, size=(h, w), mode='bilinear', align_corners=True)
                supp_out_list.append(supp_out)

        # average K foreground prototypes and K background prototypes
        # bs c -> bs c 1 1
        fg_prototypes = torch.mean(torch.cat(feature_fg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        bg_prototypes = torch.mean(torch.cat(feature_bg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)

        # measure the similarity of query features to fg/bg prototypes
        # bs 2 h w  初始化高置信度map
        similarity_0 = self.similarity_func(feature_q, fg_prototypes, bg_prototypes)

        # self-support prototype(ssp)
        # new_fg, new_bg, new_fg_local, new_bg_local
        ssfp_1, ssbp_1, asfp_1, asbp_1 = self.ssp_func(feature_q, similarity_0)
        fg_prototypes_1 = 0.5 * fg_prototypes + 0.5 * ssfp_1
        bg_prototypes_1 = 0.3 * ssbp_1 + 0.7 * asbp_1

        similarity_1 = self.similarity_func(feature_q, fg_prototypes_1, bg_prototypes_1)

        # ssp refinement
        if self.refine:
            ssfp_2, ssbp_2, asfp_2, asbp_2 = self.ssp_func(feature_q, similarity_1)
            fg_prototypes_2 = 0.5 * fg_prototypes + 0.5 * ssfp_2
            bg_prototypes_2 = 0.3 * ssbp_2 + 0.7 * asbp_2

            fg_prototypes_2 = 0.5 * fg_prototypes + 0.2 * fg_prototypes_1 + 0.3 * fg_prototypes_2
            bg_prototypes_2 = 0.5 * bg_prototypes + 0.2 * bg_prototypes_1 + 0.3 * bg_prototypes_2

            similarity_2 = self.similarity_func(feature_q, fg_prototypes_2, bg_prototypes_2)
            similarity_2 = 0.7 * similarity_2 + 0.3 * similarity_1

        similarity_1 = F.interpolate(similarity_1, size=(h, w), mode='bilinear', align_corners=True)

        if self.refine:
            similarity_2 = F.interpolate(similarity_2, size=(h, w), mode='bilinear', align_corners=True)
            out_list = [similarity_2,similarity_1]
        else:
            out_list = [similarity_1]

        if self.training:
            fg_q = self.masked_average_pooling(feature_q, (mask_q == 1).float())[None, :].squeeze(0)
            bg_q = self.masked_average_pooling(feature_q, (mask_q == 0).float())[None, :].squeeze(0)

            self_similarity_fg = F.cosine_similarity(feature_q, fg_q[..., None, None], dim=1)
            self_similarity_bg = F.cosine_similarity(feature_q, bg_q[..., None, None], dim=1)
            self_out = torch.cat((self_similarity_bg[:, None, ...], self_similarity_fg[:, None, ...]), dim=1) * 10.0

            self_out = F.interpolate(self_out, size=(h, w), mode="bilinear", align_corners=True)
            support_out = torch.cat(supp_out_list, dim=0)

            out_list.append(self_out)
            out_list.append(support_out)

        return out_list


    def masked_average_pooling(self, feature, mask):
        # feature:bs c h w   mask: bs h w
        mask = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear', align_corners=True)
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-5)
        return masked_feature  # bs c

    def similarity_func(self, feature_q, fg_proto, bg_proto):
        # feature_q: 1 c h w
        # fg_proto: 1 c 1 1
        # similarity_fg: 1 h w
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)

        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
        return out

    # feature_q：查询特征图(b c h w)  out:查询特征图与support原型逐像素相似度(b 2 h w)
    def ssp_func(self, feature_q, out):
        # feature_q: b c h w
        # out: b 2 h w
        bs, c, h, w = feature_q.shape
        # 沿着通道维数求概率值
        pred_1 = out.softmax(1)
        # b 2 h w -> b 2 h*w
        pred_1 = pred_1.view(bs, 2, -1)
        # 0通道是背景 bs c
        pred_bg = pred_1[:, 0]
        # 1通道是前景
        pred_fg = pred_1[:, 1]

        fg_list = []
        bg_list = []
        fg_local_list = []
        bg_local_list = []

        # 遍历每张查询图
        for epi in range(bs):
            # 前后背景阈值
            fg_thres = 0.7
            bg_thres = 0.6
            # 取当前张查询特征图
            cur_feat = feature_q[epi].view(c, -1)
            f_h, f_w = feature_q[epi].shape[-2:]

            # 判断当前查询特征图中是否有前景
            if (pred_fg[epi] > fg_thres).sum() > 0:
                # 有，取出那些前景的索引（每个通道上，大于阈值的索引）
                fg_feat = cur_feat[:, (pred_fg[epi] > fg_thres)]
            else:
                # 没有，取每个通道上前12最大概率值的索引
                fg_feat = cur_feat[:, torch.topk(pred_fg[epi], 12).indices]

            # 背景与前景同理
            if (pred_bg[epi] > bg_thres).sum() > 0:
                bg_feat = cur_feat[:, (pred_bg[epi] > bg_thres)]
            else:
                bg_feat = cur_feat[:, torch.topk(pred_bg[epi], 12).indices]

            # global proto
            # 将前景特征沿着最后一个维度取平均，得到每个通道的平均值 c
            fg_proto = fg_feat.mean(-1)
            bg_proto = bg_feat.mean(-1)

            # 得到全局前景原型 1 c。这里的全局前景原型主要是通过传统的query-support匹配得到，可以认为是每个通道上 初始的高置信度像素的平均
            fg_list.append(fg_proto.unsqueeze(0))  # 1 c
            bg_list.append(bg_proto.unsqueeze(0))

            # local proto
            # 相当于做一个归一化操作，这里主要是为了去构造一个[h*w c]的矩阵，代表h*w个像素，每个像素c维向量。
            # 这个c维向量就是高置信度像素的矩阵乘法
            fg_feat_norm = fg_feat / torch.norm(fg_feat, 2, 0, True)  # c n1
            bg_feat_norm = bg_feat / torch.norm(bg_feat, 2, 0, True)  # c n2
            cur_feat_norm = cur_feat / torch.norm(cur_feat, 2, 0, True)  # c h*w

            cur_feat_norm_t = cur_feat_norm.t()  # h*w c
            fg_sim = torch.matmul(cur_feat_norm_t, fg_feat_norm) * 2.0  # h*w n1
            bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0  # h*w n2

            fg_sim = fg_sim.softmax(-1)
            bg_sim = bg_sim.softmax(-1)

            fg_proto_local = torch.matmul(fg_sim, fg_feat.t())  # h*w c
            bg_proto_local = torch.matmul(bg_sim, bg_feat.t())  # h*w c

            fg_proto_local = fg_proto_local.t().view(c, f_h, f_w).unsqueeze(0)  # 1 c h w
            bg_proto_local = bg_proto_local.t().view(c, f_h, f_w).unsqueeze(0)

            fg_local_list.append(fg_proto_local)
            bg_local_list.append(bg_proto_local)

        # global proto
        new_fg = torch.cat(fg_list, dim=0).unsqueeze(-1).unsqueeze(-1) # bs c 1 1
        new_bg = torch.cat(bg_list, dim=0).unsqueeze(-1).unsqueeze(-1) # bs c 1 1

        # local proto
        new_fg_local = torch.cat(fg_local_list, dim=0).unsqueeze(-1).unsqueeze(-1)
        new_bg_local = torch.cat(bg_local_list, dim=0)

        return new_fg, new_bg, new_fg_local, new_bg_local


if __name__ == '__main__':
    model = SSPNet("resnet50",True)
    print(model)
    pass
