import sys
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def matplotlib_imshow(img, one_channel=False):
    fig = plt.figure()
    if one_channel:
        img = img.mean(dim=0)
        plt.imshow(img.numpy(), cmap="Greys")
        # plt.show()
        return img

    else:
        img = img.numpy().transpose(1, 2, 0)
        # unnorm_img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        unnorm_img = img * 255
        img = img.astype('uint8')
        unnorm_img = unnorm_img.astype('uint8')
        norm_image = torch.Tensor(img).permute(2, 0, 1)
        plt.imshow(unnorm_img)
        # plt.savefig("train_images.jpg")
        # plt.show()
        return norm_image, fig


def train_one_epoch(model, data_loader, device, optimizer, loss_function, epoch):
    model.train()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        print(
            "train epoch {} step {} train loss: {:.5f} train acc: {:.5f} lr: {:.5f}".format(
                epoch, step + 1, accu_loss.item() / (step + 1), accu_num.item() / sample_num,
                optimizer.param_groups[0]["lr"]))
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, loss_function, epoch):
    model.eval()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)

    sample_num = 0
    # data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]

        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        print("[valid epoch {} step {}] val loss: {:.5f}, val acc: {:.5f}".format(epoch, step + 1,
                                                                                  accu_loss.item() / (step + 1),
                                                                                  accu_num.item() / sample_num))
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
