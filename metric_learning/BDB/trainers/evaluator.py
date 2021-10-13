# encoding: utf-8
import numpy as np
import os
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
from loguru import logger

from trainers.re_ranking import re_ranking as re_ranking_func


class Evaluator:
    def __init__(self, model):
        self.model = model

    def save_incorrect_pairs(self, distmat, queryloader, galleryloader,
                             g_pids, q_pids, g_camids, q_camids, savefig):
        os.makedirs(savefig, exist_ok=True)
        self.model.eval()
        m = distmat.shape[0]
        indices = np.argsort(distmat, axis=1)
        for i in range(m):
            for j in range(10):
                index = indices[i][j]
                if g_camids[index] == q_camids[i] and g_pids[index] == q_pids[i]:
                    continue
                else:
                    break
            if g_pids[index] == q_pids[i]:
                continue
            fig, axes = plt.subplots(1, 11, figsize=(12, 8))
            img = queryloader.dataset.dataset[i][0]
            img = Image.open(img).convert('RGB')
            axes[0].set_title(q_pids[i])
            axes[0].imshow(img)
            axes[0].set_axis_off()
            for j in range(10):
                gallery_index = indices[i][j]
                img = galleryloader.dataset.dataset[gallery_index][0]
                img = Image.open(img).convert('RGB')
                axes[j + 1].set_title(g_pids[gallery_index])
                axes[j + 1].set_axis_off()
                axes[j + 1].imshow(img)
            import random
            n = random.randint(0, 100000)
            fig.savefig(os.path.join(savefig, '{}_{}.png'.format(q_pids[i], n)))
            plt.close(fig)

    def evaluate(self, queryloader, galleryloader, queryFliploader, galleryFliploader,
                 ranks=[1, 2, 4, 5, 8, 10, 16, 20], eval_flip=False, re_ranking=False, savefig=False):
        self.model.eval()
        qf, q_pids, q_camids = [], [], []
        for inputs0, inputs1 in zip(queryloader, queryFliploader):
            inputs, pids = self._parse_data(inputs0)
            camids = torch.ones_like(pids)
            feature0 = self._forward(inputs)
            if eval_flip:
                inputs, pids = self._parse_data(inputs1)
                camids = torch.ones_like(pids)
                feature1 = self._forward(inputs)
                qf.append((feature0 + feature1) / 2.0)
            else:
                qf.append(feature0)

            q_pids.extend(pids)
            q_camids.extend(camids)
        qf = torch.cat(qf, 0)
        q_pids = torch.Tensor(q_pids)
        q_camids = torch.Tensor(q_camids)

        logger.info("Extracted features for query set: {} x {}".format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        for inputs0, inputs1 in zip(galleryloader, galleryFliploader):
            inputs, pids = self._parse_data(inputs0)
            camids = torch.ones_like(pids)
            feature0 = self._forward(inputs)
            if eval_flip:
                inputs, pids = self._parse_data(inputs1)
                camids = torch.ones_like(pids)
                feature1 = self._forward(inputs)
                gf.append((feature0 + feature1) / 2.0)
            else:
                gf.append(feature0)

            g_pids.extend(pids)
            g_camids.extend(camids)
        gf = torch.cat(gf, 0)
        g_pids = torch.Tensor(g_pids)
        g_camids = torch.Tensor(g_camids)

        logger.info("Extracted features for gallery set: {} x {}".format(gf.size(0), gf.size(1)))

        logger.info("Computing distance matrix")

        m, n = qf.size(0), gf.size(0)
        q_g_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                   torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        q_g_dist.addmm_(1, -2, qf, gf.t())

        if re_ranking:
            q_q_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m) + \
                       torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m).t()
            q_q_dist.addmm_(1, -2, qf, qf.t())

            g_g_dist = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n) + \
                       torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n).t()
            g_g_dist.addmm_(1, -2, gf, gf.t())

            q_g_dist = q_g_dist.numpy()
            q_g_dist[q_g_dist < 0] = 0
            q_g_dist = np.sqrt(q_g_dist)

            q_q_dist = q_q_dist.numpy()
            q_q_dist[q_q_dist < 0] = 0
            q_q_dist = np.sqrt(q_q_dist)

            g_g_dist = g_g_dist.numpy()
            g_g_dist[g_g_dist < 0] = 0
            g_g_dist = np.sqrt(g_g_dist)

            distmat = torch.Tensor(re_ranking_func(q_g_dist, q_q_dist, g_g_dist))
        else:
            distmat = q_g_dist

        if savefig:
            logger.info("Saving fingure")
            self.save_incorrect_pairs(distmat.numpy(), queryloader, galleryloader,
                                      g_pids.numpy(), q_pids.numpy(), g_camids.numpy(), q_camids.numpy(), savefig)

        logger.info("Computing CMC and mAP")
        cmc, mAP = self.eval_func_gpu(distmat, q_pids, g_pids, q_camids, g_camids)

        logger.info("Results ----------")
        logger.info("mAP: {:.1%}".format(mAP))
        logger.info("CMC curve")
        for r in ranks:
            logger.info("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        logger.info("------------------")

        return cmc[0]

    def _parse_data(self, inputs):
        imgs, pids = inputs
        return imgs.cuda(), pids.cuda()

    def _forward(self, inputs):
        with torch.no_grad():
            feature = self.model(inputs)
        return feature.cpu()

    def eval_func_gpu(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        num_q, num_g = distmat.size()
        if num_g < max_rank:
            max_rank = num_g
            logger.info("Note: number of gallery samples is quite small, got {}".format(num_g))
        _, indices = torch.sort(distmat, dim=1)
        matches = g_pids[indices] == q_pids.view([num_q, -1])
        keep = ~((g_pids[indices] == q_pids.view([num_q, -1])) & (g_camids[indices] == q_camids.view([num_q, -1])))
        # keep = g_camids[indices]  != q_camids.view([num_q, -1])
        # keep = (g_pids[indices] == q_pids.view([num_q, -1]))

        results = []
        num_rel = []
        for i in range(num_q):
            # m = matches[i][keep[i]]
            m = matches[i]
            if m.any():
                num_rel.append(m.sum())
                results.append(m[:max_rank].unsqueeze(0))
        matches = torch.cat(results, dim=0).float()
        num_rel = torch.Tensor(num_rel)

        cmc = matches.cumsum(dim=1)
        cmc[cmc > 1] = 1
        all_cmc = cmc.sum(dim=0) / cmc.size(0)

        pos = torch.Tensor(range(1, max_rank + 1))
        temp_cmc = matches.cumsum(dim=1) / pos * matches
        AP = temp_cmc.sum(dim=1) / num_rel
        mAP = AP.sum() / AP.size(0)
        return all_cmc.numpy(), mAP.item()

    def eval_func(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
        """Evaluation with market1501 metric
            Key: for each query identity, its gallery images from the same camera view are discarded.
            """
        num_q, num_g = distmat.shape
        if num_g < max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        indices = np.argsort(distmat, axis=1)
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        num_valid_q = 0.  # number of valid query
        for q_idx in range(num_q):
            # get query pid and camid
            q_pid = q_pids[q_idx]
            q_camid = q_camids[q_idx]

            # remove gallery samples that have the same pid and camid with query
            order = indices[q_idx]
            remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
            keep = np.invert(remove)

            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            orig_cmc = matches[q_idx][keep]
            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = orig_cmc.sum()
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        return all_cmc, mAP


def test(model, queryloader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target, _ in queryloader:
            output = model(data).cpu()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    rank1 = 100. * correct / len(queryloader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(queryloader.dataset), rank1))
    return rank1

# # encoding: utf-8
# import numpy as np
# import os
# import torch
# from PIL import Image
# import matplotlib.pyplot as plt
#
# from trainers.re_ranking import re_ranking as re_ranking_func
#
# class ResNetEvaluator:
#     def __init__(self, model):
#         self.model = model
#
#     def save_incorrect_pairs(self, distmat, queryloader, galleryloader,
#         g_pids, q_pids, g_camids, q_camids, savefig):
#         os.makedirs(savefig, exist_ok=True)
#         self.model.eval()
#         m = distmat.shape[0]
#         indices = np.argsort(distmat, axis=1)
#         for i in range(m):
#             for j in range(10):
#                 index = indices[i][j]
#                 if g_camids[index] == q_camids[i] and g_pids[index] == q_pids[i]:
#                     continue
#                 else:
#                     break
#             if g_pids[index] == q_pids[i]:
#                 continue
#             fig, axes =plt.subplots(1, 11, figsize=(12, 8))
#             img = queryloader.dataset.dataset[i][0]
#             img = Image.open(img).convert('RGB')
#             axes[0].set_title(q_pids[i])
#             axes[0].imshow(img)
#             axes[0].set_axis_off()
#             for j in range(10):
#                 gallery_index = indices[i][j]
#                 img = galleryloader.dataset.dataset[gallery_index][0]
#                 img = Image.open(img).convert('RGB')
#                 axes[j+1].set_title(g_pids[gallery_index])
#                 axes[j+1].set_axis_off()
#                 axes[j+1].imshow(img)
#             fig.savefig(os.path.join(savefig, '%d.png' %q_pids[i]))
#             plt.close(fig)
#
#     def evaluate(self, queryloader, galleryloader, queryFliploader, galleryFliploader,
#         ranks=[1, 2, 4, 5,8, 10, 16, 20], eval_flip=False, re_ranking=False, savefig=False):
#         self.model.eval()
#         qf, q_pids, q_camids = [], [], []
#         for inputs0, inputs1 in zip(queryloader, queryFliploader):
#             inputs, pids = self._parse_data(inputs0)
#             camids = torch.ones_like(pids)
#             feature0 = self._forward(inputs)
#             if eval_flip:
#                 inputs, pids = self._parse_data(inputs1)
#                 camids = torch.ones_like(pids)
#                 feature1 = self._forward(inputs)
#                 qf.append((feature0 + feature1) / 2.0)
#             else:
#                 qf.append(feature0)
#
#             q_pids.extend(pids)
#             q_camids.extend(camids)
#         qf = torch.cat(qf, 0)
#         q_pids = torch.Tensor(q_pids)
#         q_camids = torch.Tensor(q_camids)
#
#         print("Extracted features for query set: {} x {}".format(qf.size(0), qf.size(1)))
#
#         gf, g_pids, g_camids = [], [], []
#         for inputs0, inputs1 in zip(galleryloader, galleryFliploader):
#             inputs, pids = self._parse_data(inputs0)
#             camids = torch.ones_like(pids)
#             feature0 = self._forward(inputs)
#             if eval_flip:
#                 inputs, pids = self._parse_data(inputs1)
#                 feature1 = self._forward(inputs)
#                 camids = torch.ones_like(pids)
#                 gf.append((feature0 + feature1) / 2.0)
#             else:
#                 gf.append(feature0)
#
#             g_pids.extend(pids)
#             g_camids.extend(camids)
#         gf = torch.cat(gf, 0)
#         g_pids = torch.Tensor(g_pids)
#         g_camids = torch.Tensor(g_camids)
#
#         print("Extracted features for gallery set: {} x {}".format(gf.size(0), gf.size(1)))
#
#         print("Computing distance matrix")
#
#         m, n = qf.size(0), gf.size(0)
#         q_g_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#             torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#         q_g_dist.addmm_(1, -2, qf, gf.t())
#
#         if re_ranking:
#             q_q_dist = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m) + \
#                 torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, m).t()
#             q_q_dist.addmm_(1, -2, qf, qf.t())
#
#             g_g_dist = torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n) + \
#                 torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, n).t()
#             g_g_dist.addmm_(1, -2, gf, gf.t())
#
#             q_g_dist = q_g_dist.numpy()
#             q_g_dist[q_g_dist < 0] = 0
#             q_g_dist = np.sqrt(q_g_dist)
#
#             q_q_dist = q_q_dist.numpy()
#             q_q_dist[q_q_dist < 0] = 0
#             q_q_dist = np.sqrt(q_q_dist)
#
#             g_g_dist = g_g_dist.numpy()
#             g_g_dist[g_g_dist < 0] = 0
#             g_g_dist = np.sqrt(g_g_dist)
#
#             distmat = torch.Tensor(re_ranking_func(q_g_dist, q_q_dist, g_g_dist))
#         else:
#             distmat = q_g_dist
#
#         if savefig:
#             print("Saving fingure")
#             self.save_incorrect_pairs(distmat.numpy(), queryloader, galleryloader,
#                 g_pids.numpy(), q_pids.numpy(), g_camids.numpy(), q_camids.numpy(), savefig)
#
#         print("Computing CMC and mAP")
#         cmc, mAP = self.eval_func_gpu(distmat, q_pids, g_pids, q_camids, g_camids)
#
#         print("Results ----------")
#         print("mAP: {:.1%}".format(mAP))
#         print("CMC curve")
#         for r in ranks:
#             print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
#         print("------------------")
#
#         return cmc[0]
#
#     def _parse_data(self, inputs):
#         imgs, pids = inputs
#         return imgs.cuda(), pids
#
#     def _forward(self, inputs):
#         with torch.no_grad():
#             feature = self.model(inputs)
#         return feature.cpu()
#
#     def eval_func_gpu(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
#         num_q, num_g = distmat.size()
#         if num_g < max_rank:
#             max_rank = num_g
#             print("Note: number of gallery samples is quite small, got {}".format(num_g))
#         _, indices = torch.sort(distmat, dim=1)
#         matches = g_pids[indices] == q_pids.view([num_q, -1])
#         keep = ~((g_pids[indices] == q_pids.view([num_q, -1])) & (g_camids[indices]  == q_camids.view([num_q, -1])))
#         #keep = g_camids[indices]  != q_camids.view([num_q, -1])
#
#         results = []
#         num_rel = []
#         for i in range(num_q):
#             #m = matches[i][keep[i]]
#             m = matches[i]
#             if m.any():
#                 num_rel.append(m.sum())
#                 results.append(m[:max_rank].unsqueeze(0))
#         matches = torch.cat(results, dim=0).float()
#         num_rel = torch.Tensor(num_rel)
#
#         cmc = matches.cumsum(dim=1)
#         cmc[cmc > 1] = 1
#         all_cmc = cmc.sum(dim=0) / cmc.size(0)
#
#         pos = torch.Tensor(range(1, max_rank+1))
#         temp_cmc = matches.cumsum(dim=1) / pos * matches
#         AP = temp_cmc.sum(dim=1) / num_rel
#         mAP = AP.sum() / AP.size(0)
#         return all_cmc.numpy(), mAP.item()
#
#     def eval_func(self, distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
#         """Evaluation with market1501 metric
#             Key: for each query identity, its gallery images from the same camera view are discarded.
#             """
#         num_q, num_g = distmat.shape
#         if num_g < max_rank:
#             max_rank = num_g
#             print("Note: number of gallery samples is quite small, got {}".format(num_g))
#         indices = np.argsort(distmat, axis=1)
#         matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
#
#         # compute cmc curve for each query
#         all_cmc = []
#         all_AP = []
#         num_valid_q = 0.  # number of valid query
#         for q_idx in range(num_q):
#             # get query pid and camid
#             q_pid = q_pids[q_idx]
#             q_camid = q_camids[q_idx]
#
#             # remove gallery samples that have the same pid and camid with query
#             order = indices[q_idx]
#             remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
#             keep = np.invert(remove)
#
#             # compute cmc curve
#             # binary vector, positions with value 1 are correct matches
#             orig_cmc = matches[q_idx][keep]
#             if not np.any(orig_cmc):
#                 # this condition is true when query identity does not appear in gallery
#                 continue
#
#             cmc = orig_cmc.cumsum()
#             cmc[cmc > 1] = 1
#
#             all_cmc.append(cmc[:max_rank])
#             num_valid_q += 1.
#
#             # compute average precision
#             # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
#             num_rel = orig_cmc.sum()
#             tmp_cmc = orig_cmc.cumsum()
#             tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
#             tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
#             AP = tmp_cmc.sum() / num_rel
#             all_AP.append(AP)
#
#         assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
#
#         all_cmc = np.asarray(all_cmc).astype(np.float32)
#         all_cmc = all_cmc.sum(0) / num_valid_q
#         mAP = np.mean(all_AP)
#
#         return all_cmc, mAP
