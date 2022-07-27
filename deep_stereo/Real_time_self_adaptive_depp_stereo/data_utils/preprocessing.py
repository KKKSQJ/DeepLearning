import PIL.ImageMath
import matplotlib.pyplot as plt
import numpy as np
import random

import torch
from PIL import Image
import torchvision

FULLY_DIFFERENTIABLE = False


def random_crop(image, crop_shape, padding=None):
    if not isinstance(image, list):
        image = [image]
    oshape = np.shape(image[0])
    assert oshape[0] - crop_shape[0] > 0
    assert oshape[1] - crop_shape[1] > 0
    out = []

    nh = random.randint(0, oshape[0] - crop_shape[0])
    nw = random.randint(0, oshape[1] - crop_shape[1])
    for img in image:
        image_crop = img[nh:nh + crop_shape[0], nw:nw + crop_shape[1]]
        if padding:
            # oshape = (image_crop.shape[0] + 2 * padding, image_crop.shape[1] + 2 * padding)
            npad = ((padding, padding), (padding, padding), (0, 0))
            image_crop = np.lib.pad(image_crop, pad_width=npad, mode='constant', constant_values=0)
        out.append(image_crop)
    return out


def resize_image_with_crop_or_pad(image, target_h, target_w):
    assert target_h > 0
    assert target_w > 0
    h, w, c = np.shape(image)
    if h < target_h:
        padding = target_h - h
        npad = ((0, padding), (0, 0), (0, 0))
        image = np.lib.pad(image, pad_width=npad, mode='constant', constant_values=0)
    else:
        #image = image[0:target_h, :, :]
        image = image[int((h - target_h) / 2):int((h + target_h) / 2), :, :]
    if w < target_w:
        padding = target_w - w
        npad = ((0, 0), (0, padding), (0, 0))
        image = np.lib.pad(image, pad_width=npad, mode='constant', constant_values=0)
    else:
        # image = image[:, 0:target_w, :]
        image = image[:, int((w - target_w) / 2):int((w + target_w) / 2), :]
    return image


def augment(left_image, right_image):
    pass


def pad_image(image, down_factor=256):
    image_shape = image.shape
    new_height = (torch.floor_divide(image_shape[2], down_factor) + 1) * down_factor if image_shape[
                                                                                            2] % down_factor != 0 else \
        image_shape[2]
    new_width = (torch.floor_divide(image_shape[3], down_factor) + 1) * down_factor if image_shape[
                                                                                           3] % down_factor != 0 else \
        image_shape[3]
    pad_height_left = (new_height - image_shape[2]) // 2
    pad_height_right = (new_height - image_shape[2] + 1) // 2
    pad_width_left = (new_width - image_shape[3]) // 2
    pad_width_right = (new_width - image_shape[3] + 1) // 2

    image = torch.nn.functional.pad(image, (pad_width_left, pad_width_right,
                                            pad_height_left, pad_height_right), mode="reflect")
    return image


def _rescale_torch(img, out_shape):
    pass


def rescale_image(image, out_shape):
    if FULLY_DIFFERENTIABLE:
        return _rescale_torch(img, out_shape)
    else:
        resize = torchvision.transforms.Resize(list(out_shape))
        return resize(image)


def resize_to_prediction(x, pred):
    return rescale_image(x, pred.shape[2:4])


def bilinear_sampler(imgs, coords):
    """
    Construct a new image by bilinear sampling from the input image.
    Points falling outside the source image boundary have value 0.
    Args:
        imgs: source image to be sampled from [batch, channels, height_s, width_s] to [batch, height_s, width_s, channels]
        coords: coordinates of source pixels to sample from [batch, 2, height_t,width_t] to [batch, height_t,width_t, 2]. height_t/width_t correspond to the dimensions of the outputimage (don't need to be the same as height_s/width_s). The two channels correspond to x and y coordinates respectively.
    Returns:
        A new sampled image [batch, height_t, width_t, channels] to [batch, channels, height_t, width_t]
    """
    imgs = imgs.permute(0, 2, 3, 1).contiguous()
    coords = coords.permute(0, 2, 3, 1).contiguous()
    imgs = imgs.to(coords.device)

    def _repeat(x, n_repeats):
        rep = torch.unsqueeze(torch.ones(n_repeats), dim=0)
        x = torch.matmul(torch.reshape(x, [-1, 1]), rep)
        return torch.reshape(x, [-1])

    coords_x, coords_y = torch.split(coords, [1, 1], dim=3)
    inp_size = imgs.shape
    coords_size = coords.shape
    out_size = [coords_size[0], coords_size[1], coords_size[2], inp_size[3]]

    coords_x = coords_x.float()
    coords_y = coords_y.float()

    x0 = torch.floor(coords_x)
    x1 = x0 + 1
    y0 = torch.floor(coords_y)
    y1 = y0 + 1

    y_max = float(inp_size[1] - 1)
    x_max = float(inp_size[2] - 1)
    zero = torch.zeros([1], dtype=torch.float32)

    wt_x0 = x1 - coords_x
    wt_x1 = coords_x - x0
    wt_y0 = y1 - coords_y
    wt_y1 = coords_y - y0

    x0_safe = torch.clamp(x0, zero[0], x_max)
    y0_safe = torch.clamp(y0, zero[0], y_max)
    x1_safe = torch.clamp(x1, zero[0], x_max)
    y1_safe = torch.clamp(y1, zero[0], y_max)

    dim2 = float(inp_size[2])
    dim1 = float(inp_size[2] * inp_size[1])
    base = torch.reshape(
        _repeat(torch.arange(0, coords_size[0], dtype=torch.float32) * dim1, coords_size[1] * coords_size[2]),
        [out_size[0], out_size[1], out_size[2], 1])
    base = base.to(coords.device)

    base_y0 = base + y0_safe * dim2
    base_y1 = base + y1_safe * dim2
    idx00 = x0_safe + base_y0
    idx01 = x0_safe + base_y1
    idx10 = x1_safe + base_y0
    idx11 = x1_safe + base_y1

    idx00 = torch.reshape(idx00, (-1, 1)).expand(coords_size[1] * coords_size[2], inp_size[3])
    idx01 = torch.reshape(idx01, (-1, 1)).expand(coords_size[1] * coords_size[2], inp_size[3])
    idx10 = torch.reshape(idx10, (-1, 1)).expand(coords_size[1] * coords_size[2], inp_size[3])
    idx11 = torch.reshape(idx11, (-1, 1)).expand(coords_size[1] * coords_size[2], inp_size[3])

    imgs_flat = torch.reshape(imgs, (-1, inp_size[3]))
    im00 = torch.reshape(torch.gather(imgs_flat, 0, idx00.long()), out_size)
    im01 = torch.reshape(torch.gather(imgs_flat, 0, idx01.long()), out_size)
    im10 = torch.reshape(torch.gather(imgs_flat, 0, idx10.long()), out_size)
    im11 = torch.reshape(torch.gather(imgs_flat, 0, idx11.long()), out_size)

    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1

    output = w00 * im00.add(w01 * im01).add(w10 * im10).add(w11 * im11)
    output = output.permute(0, 3, 1, 2).contiguous()
    return output


def warp_image(img, flow):
    """
    Given an image and a flow generate the warped image, for stereo img is the right image, flow is the disparity alligned with left
    img: image that needs to be warped
    flow: Generic optical flow or disparity
    """

    def build_coords(immy):
        max_height = 2048
        max_width = 2048
        # pixel_coords = np.ones((1, 2, max_height, max_width))
        # # build pixel coordinates and their disparity
        # for i in range(0, max_height):
        #     for j in range(0, max_width):
        #         pixel_coords[0][0][i][j] = j
        #         pixel_coords[0][0][i][j] = i
        yv, xv = torch.meshgrid([torch.arange(max_height), torch.arange(max_width)])
        pixel_coords = torch.stack((xv, yv), 0).view(1, 2, max_height, max_width).type(torch.float32)

        # pixel_coords = torch.tensor(pixel_coords, dtype=torch.float32)
        real_height = immy.shape[2]
        real_width = immy.shape[3]
        real_pixel_coords = pixel_coords[:, :, 0:real_height, 0:real_width]
        real_pixel_coords = real_pixel_coords.to(immy.device)
        immy = torch.cat([immy, torch.zeros_like(immy)], dim=1)
        output = real_pixel_coords - immy
        return output

    coords = build_coords(flow)
    warped = bilinear_sampler(img, coords)
    return warped


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


if __name__ == '__main__':
    # file = r'E:\pic\2.jpg'
    # img = Image.open(file)
    # img = np.array(img)
    # img = img.transpose((2, 0, 1))
    # img = torch.Tensor(img)
    # img = img.unsqueeze(0)
    # x = rescale_image(img, (20, 20))
    # x = pad_image(img, 64)
    #
    # b = random_crop(img, (150, 150))
    # c = resize_image_with_crop_or_pad(img, 150, 150)
    # d = resize_image_with_crop_or_pad(img, 3000, 3000)
    # a = torch.randn(1, 3, 7, 7)
    # b = torch.randn(1, 2, 5, 5)
    # c = bilinear_sampler(a, b)

    a = torch.range(0, 1)
    b = torch.arange(0, 1, dtype=torch.float32)

    print(1)
