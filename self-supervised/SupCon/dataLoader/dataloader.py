from torch.utils.data import DataLoader

from dataLoader.datasets import create_supcon_dataset
from dataLoader.transforms import TwoCropTransform


def build_loaders(data_dir, transforms, batch_sizes, num_workers, second_stage=False):
    dataset_name = data_dir.split('/')[-1]

    if second_stage:
        train_features_dataset = create_supcon_dataset(dataset_name, data_dir=data_dir, train=True,
                                                       transform=transforms['train_transforms'], second_stage=True)
    else:
        # train_features_dataset is used for evaluation -> hence, we don't need TwoCropTransform
        train_features_dataset = create_supcon_dataset(dataset_name, data_dir=data_dir, train=True,
                                                       transform=transforms['valid_transforms'], second_stage=True)

        train_supcon_dataset = create_supcon_dataset(dataset_name, data_dir=data_dir, train=True,
                                                     transform=TwoCropTransform(transforms['train_transforms']),
                                                     second_stage=False)

    valid_dataset = create_supcon_dataset(dataset_name, data_dir=data_dir, train=False,
                                          transform=transforms['valid_transforms'], second_stage=True)

    if not second_stage:
        train_supcon_loader = DataLoader(
            train_supcon_dataset, batch_size=batch_sizes['train_batch_size'], shuffle=True,
            num_workers=num_workers, pin_memory=True)

    train_features_loader = DataLoader(
        train_features_dataset, batch_size=batch_sizes['train_batch_size'], shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)

    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_sizes['valid_batch_size'], shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)

    if second_stage:
        return {'train_features_loader': train_features_loader, 'valid_loader': valid_loader}

    return {'train_supcon_loader': train_supcon_loader, 'train_features_loader': train_features_loader,
            'valid_loader': valid_loader}
