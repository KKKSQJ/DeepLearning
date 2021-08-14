import sys
from tqdm import tqdm
import torch

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    acc_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_class = torch.max(pred, dim=1)[1]
        acc_num += torch.eq(pred_class, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)   # 更新平均loss

        # 打印平均loss
        data_loader.desc = "[train epoch {}] train loss: {}, train acc: {:.3f}".format(epoch, round(mean_loss.item(), 3), acc_num.item() / sample_num)  # round(value, 3):表示去小数点前3位，4舍5入
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        # 更新优化器参数，优化器参数根据反向传播的梯度来更新
        optimizer.step()
        # 每个mini-batch之后，需要清0。因为我们假定一次mini-batch就是一个训练集
        optimizer.zero_grad()


    return mean_loss.item(), acc_num.item()/sample_num

# 修饰器，等价于with torch.no_grad() 即不计算梯度，不进行反向传播，一般用于验证和测试阶段
@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    # 统计验证集样本总数目
    num_samples = len(data_loader.dataset)

    # 打印验证进度
    data_loader = tqdm(data_loader, desc="validation...")

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        # torch.max(x,dim=1): 取第二个维度的最大值，返回[0]:概率值。[1]：对应类别的索引
        pred_class = torch.max(pred, dim=1)[1]
        # eq: ==
        accu_num += torch.eq(pred_class, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] val loss: {:.3f}, val acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / num_samples)

    # 计算平均损失
    loss = accu_loss.item() / (step+1)
    # 计算预测正确的比例
    acc = accu_num.item() / num_samples

    return loss, acc