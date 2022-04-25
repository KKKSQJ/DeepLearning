import os

from MulticoreTSNE import MulticoreTSNE as TSNE
from matplotlib import pyplot as plt
from dataLoader.datasets import *
from dataLoader.dataloader import *
from dataLoader.transforms import *
from models.model import *
import torch
import argparse
from trainer.trainer import compute_embeddings


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_pretrained", type=str, default="weights/supcon_first_stage_cifar10/swa", )
    parser.add_argument("--data_dir", type=str, default="data/cifar10")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--stage", type=str, default="first")
    parser.add_argument("--show", type=bool, default=True)

    args = parser.parse_args()
    return args


def run(ckpt_pretrained="weights/supcon_first_stage_cifar10/swa",
        data_dir="data/cifar10",
        num_classes=10,
        backbone="resnet18",
        stage="first",
        show=True,
        ):

    assert os.path.exists(ckpt_pretrained)
    assert os.path.exists(data_dir)

    scaler = torch.cuda.amp.GradScaler()

    batch_sizes = {
        "train_batch_size": 20,
        'valid_batch_size': 20
    }

    num_workers = 16

    transforms = build_transforms(second_stage=(stage == 'second'))
    loaders = build_loaders(data_dir, transforms, batch_sizes, num_workers, second_stage=(stage == 'second'))

    model = build_model(backbone, second_stage=(stage == 'second'), num_classes=num_classes,
                        ckpt_pretrained=ckpt_pretrained).cuda()
    model.use_projection_head(False)
    model.eval()

    embeddings, labels = compute_embeddings(loaders['valid_loader'], model, scaler)

    embeddings_tsne = TSNE(n_jobs=num_workers).fit_transform(embeddings)
    vis_x = embeddings_tsne[:, 0]
    vis_y = embeddings_tsne[:, 1]

    if show:
        plt.scatter(vis_x, vis_y, c=labels, cmap=plt.cm.get_cmap("jet", num_classes), marker='.')
        plt.colorbar(ticks=range(num_classes))
        plt.show()

    embeddings, labels = compute_embeddings(loaders['train_features_loader'], model, scaler)
    embeddings_tsne = TSNE(n_jobs=num_workers).fit_transform(embeddings)
    vis_x = embeddings_tsne[:, 0]
    vis_y = embeddings_tsne[:, 1]

    if show:
        plt.scatter(vis_x, vis_y, c=labels, cmap=plt.cm.get_cmap("jet", num_classes), marker='.')
        plt.colorbar(ticks=range(num_classes))
        plt.show()


if __name__ == '__main__':
    args = parser_args()
    run(**vars(args))
