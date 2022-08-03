import torch.nn.functional as F
import torch
import numpy as np


# 自相关
def stereo_cost_volume_correlation(reference, target, radius_x=2, stride=1):
    cost_curve = correlation(reference, target, radius_x, stride)
    cost_curve = torch.cat([reference, cost_curve], dim=1)
    return cost_curve


def correlation(reference, target, radius_x=2, stride=1):
    cost_curve = []
    target_shape = target.shape
    target_feature = F.pad(target, [radius_x, radius_x], value=0)
    for start, i in enumerate(range(-radius_x, radius_x + 1, stride)):
        shifted = target_feature[..., i + radius_x:start + target_shape[-1]]
        cost_curve.append(torch.mean(shifted * reference, dim=1, keepdim=True))
    result = torch.cat(cost_curve, dim=1)
    return result


def gather_nd_torch(params, indices):
    params = torch.movedim(params, (0, 1, 2, 3), (0, 3, 1, 2))
    indices = torch.movedim(indices, (0, 1, 2, 3), (0, 3, 1, 2))
    indices = indices.type(torch.long)
    gathered = params[list(indices.T)]
    gathered = torch.movedim(gathered, (0, 1, 2, 3), (3, 2, 0, 1))
    return gathered


def resize_image_with_crop_or_pad(image, target_h, target_w):
    assert target_h > 0
    assert target_w > 0
    requires_grad = image.requires_grad
    device = image.device
    image = image.squeeze(0)
    image = image.detach().cpu().numpy()
    image = image.transpose((1, 2, 0))
    h, w, c = np.shape(image)
    if h < target_h:
        padding = target_h - h
        npad = ((0, padding), (0, 0), (0, 0))
        image = np.lib.pad(image, pad_width=npad, mode='constant', constant_values=0)
    else:
        # image = image[0:target_h, :, :]
        image = image[:, int((w - target_w) / 2):int((w + target_w) / 2), :]
    if w < target_w:
        padding = target_w - w
        npad = ((0, 0), (0, padding), (0, 0))
        image = np.lib.pad(image, pad_width=npad, mode='constant', constant_values=0)
    else:
        # image = image[:, 0:target_w, :]
        image = image[int((h - target_h) / 2):int((h + target_h) / 2), :, :]
    image = image.transpose((2, 0, 1))
    image = torch.tensor(image, requires_grad=requires_grad)
    image = image.unsqueeze(0).to(device)
    return image


def resize_image_with_crop_or_pad_2(image, target_h, target_w):
    assert target_h > 0
    assert target_w > 0

    image = image.squeeze(0)
    if image.shape[1] < target_h:
        padding = target_h - image.shape[1]
        npad = (0, 0, 0, padding, 0, 0)
        image = F.pad(image, npad, mode='constant', value=0)
    else:
        image = image[:, int((image.shape[1] - target_h) / 2):int((image.shape[1] + target_h) / 2), :]
    if image.shape[-1] < target_w:
        padding = target_w - image.shape[-1]
        npad = (0, 0, 0, 0, 0, padding)
        image = F.pad(image, npad, mode='constant', value=0)
    else:
        image = image[:, :, int((image.shape[-1] - target_w) / 2):int((image.shape[-1] + target_w) / 2)]
    image = image.unsqueeze(0)
    return image


if __name__ == '__main__':
    pixel_coords = np.ones((1, 2, 3, 5))
    for i in range(0, 3):
        for j in range(0, 5):
            pixel_coords[0][0][i][j] = j
            pixel_coords[0][1][i][j] = i

    yv, xv = torch.meshgrid([torch.arange(3), torch.arange(5)])
    grid = torch.stack((xv, yv), 0).view(1, 2, 3, 5).type(torch.float32)
    print(1)

    a = torch.randn(1, 3, 5, 5)
    x = resize_image_with_crop_or_pad(a, 4, 4)
    print(1)
