import albumentations as A
from albumentations import pytorch as AT


def build_transforms(second_stage):
    if second_stage:
        train_transforms = A.Compose([
            # A.Flip(),
            # A.Rotate(),
            A.Resize(224, 224),
            A.Normalize(),
            AT.ToTensorV2()
        ])
        valid_transforms = A.Compose([A.Resize(224, 224), A.Normalize(), AT.ToTensorV2()])

        transforms_dict = {
            "train_transforms": train_transforms,
            "valid_transforms": valid_transforms,
        }
    else:
        train_transforms = A.Compose([
            A.RandomResizedCrop(height=224, width=224, scale=(0.15, 1.)),
            A.Rotate(),
            A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.9),
            A.ToGray(p=0.2),
            A.Normalize(),
            AT.ToTensorV2(),
        ])

        valid_transforms = A.Compose([A.Resize(224, 224), A.Normalize(), AT.ToTensorV2()])

        transforms_dict = {
            "train_transforms": train_transforms,
            'valid_transforms': valid_transforms,
        }

    return transforms_dict


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, crop_transform):
        self.crop_transform = crop_transform

    def __call__(self, x):
        return [self.crop_transform(image=x), self.crop_transform(image=x)]
