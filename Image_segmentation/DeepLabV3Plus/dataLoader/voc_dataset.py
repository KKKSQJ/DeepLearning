import os
from dataLoader.base_dataset import BasicDataset


class VOCSegmentation(BasicDataset):
    def __init__(self,
                 voc_root,
                 year="2012",
                 transforms=None,
                 txt_name: str = "train.txt",
                 label_mapping=None,
                 ):
        assert year in ["2007", "2012"], "year must be in ['2007', '2012']"
        root = os.path.join(voc_root, "VOCdevkit", f"VOC{year}")
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir = os.path.join(root, 'SegmentationClass')

        txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
        assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        with open(os.path.join(txt_path), "r") as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        super(VOCSegmentation, self).__init__(image_dir, mask_dir, file_names, transforms, label_mapping)

# if __name__ == '__main__':
#     import dataLoader.transforms as T
#     import matplotlib.pyplot as plt
#     import numpy as np
#
#     pallette = []
#     color = {"0": [0, 0, 255], "1": [128, 0, 0], "255": [255, 255, 255]}
#     for i in color.values():
#         pallette += i
#
#     images_dir = r'E:\dataset\segmentation\car\train_images'
#     masks_dir = r'E:\dataset\segmentation\car\train_masks'
#     trans = T.Compose(
#         [
#             T.RandomResize(200, 300),
#             T.RandomHorizontalFlip(1.0),
#             T.RandomCrop(250),
#             # T.GaussianBlur(11, 2),
#             # T.ColorJitter(),
#             # T.ToTensor(),
#             # T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
#         ]
#     )
#     # d = BasicDataset(images_dir, masks_dir, transforms=trans)
#     d = VOCSegmentation(voc_root=r'E:\dataset\VOC2012', transforms=trans)
#     for b in d:
#         # b = d[0]
#         image = b[0]
#         mask = b[1]
#         plt.subplot(1, 2, 1)
#         plt.imshow(image)
#         plt.title("image")
#
#         plt.subplot(1, 2, 2)
#         mask.putpalette(pallette)
#         plt.imshow(mask)
#         plt.title("mask")
#         plt.ion()
#         plt.pause(0.01)
#         plt.waitforbuttonpress()
#         plt.show()
