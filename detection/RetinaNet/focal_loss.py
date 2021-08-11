import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self,gamma=2.0,alpha=0.25,reduction='mean'):
        super(FocalLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self,pred,label):
        # FL = - self.alpha * (1.0000001 - p_t) ** self.gamma * log(p_t)
        # log(p_t) = nn.BCEWithLogitsLoss()
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        """
        3样本，2分类 label=0或者1
        input = torch.randn(3,requires_grad=True)
        target = torch.FloatTensor([1,1,0])
        out = FocalLoss(input,target)
        """
        loss = self.loss_fcn(pred, label)
        pred_prob = torch.sigmoid(pred)
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss






if __name__ == '__main__':
    BCEloss = nn.BCEWithLogitsLoss()
    FL = FocalLoss(gamma=1.5)
    FL2 = FocalLoss()
    # 3个样本2分类
    input = torch.randn(3,requires_grad=True)
    print("input: {}".format(input))
    #target = torch.empty(3).random_(2)
    target = torch.FloatTensor([1,1,0])
    print("target: {}".format(target))
    BCEout = BCEloss(input,target)
    print("BCE LOSS: {}".format(BCEout))
    FLout = FL(input,target)
    print("FL: {}".format(FLout))
    FLout2 = FL2(input,target)
    print("FL2: {}".format(FLout2))

    m = nn.Sigmoid()
    sigmoid = m(input)
    """
    bce loss = 1/n * abs(sum( yn*ln(xn) + (1-yn)*ln(1-xn)))  
    n:样本个数,是否除以n，取决于reduction，默认='mean'
    yn：二分类，0或者1
    xn:网络预测值经过sigmoid后的值
    """

    print("sigmoid: {}".format(sigmoid))
