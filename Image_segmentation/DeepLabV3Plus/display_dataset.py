from dataLoader.transforms import build_transform
from dataLoader.dataloader import build_dataloader
import matplotlib.pyplot as plt
from PIL import Image


def show_dataset(cfg='config/example.yaml'):
    transforms_train = build_transform(cfg=cfg, train=True)
    transforms_val = build_transform(cfg=cfg, train=False)
    transforms = {"train": transforms_train, "val": transforms_val}
    dataloader = build_dataloader(cfg=cfg, transform=transforms)
    train_loader = dataloader["train"]
    val_loader = dataloader["val"]

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    pallette = []
    color = {"0": [0, 0, 255], "1": [128, 0, 0], "255": [255, 255, 255]}
    for i in color.values():
        pallette += i

    for i, data in enumerate(train_loader):
        image = data[0][0].numpy()
        mask = data[1][0].numpy()

        image = image.transpose((1, 2, 0))
        image = image * std + mean
        image = image * 255.0
        image = image.astype('uint8')[:, :, ::-1]
        mask = mask.astype('uint8')
        mask = Image.fromarray(mask)

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("image")

        plt.subplot(1, 2, 2)
        mask.putpalette(pallette)
        plt.imshow(mask)
        plt.title("mask")
        plt.ion()
        plt.pause(0.01)
        plt.waitforbuttonpress()
        plt.show()


if __name__ == '__main__':
    show_dataset(cfg='config/example.yaml')
