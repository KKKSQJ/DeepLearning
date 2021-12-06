import torch
import matplotlib.pyplot as plt


def matplotlib_imshow(img, one_channel=False):
    fig = plt.figure()
    if one_channel:
        img = img.mean(dim=0)
        plt.imshow(img.numpy(), cmap="Greys")
        # plt.show()
        return img

    else:
        img = img.numpy().transpose(1, 2, 0)
        unnorm_img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        #unnorm_img = img * 255
        img = img.astype('uint8')
        unnorm_img = unnorm_img.astype('uint8')
        norm_image = torch.Tensor(img).permute(2, 0, 1)
        plt.imshow(unnorm_img)
        # plt.savefig("train_images.jpg")
        # plt.show()
        return norm_image, fig
