import torch.nn as nn
import torch
from torchsummary import summary
from torchvision.models import AlexNet, resnet50


class mnist_cnn(nn.Module):
    def __init__(self, num_classes):
        super(mnist_cnn, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 32x28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x14x14
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64x14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x7x7
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 64x7x7
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x3x3
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 3 * 3, 128),  # 64x3x3->128
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)  # 等价于x.view(x.size(0),-1)
        x = self.fc(x)
        return x


class mnist_fcn(nn.Module):
    def __init__(self, num_classes):
        super(mnist_fcn, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 3x28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 32x14x14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64x14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 64x7x7
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # 64x7x7
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 64x3x3
        )
        # 用卷积层代替全连接层，卷积核大小等于上一层的feature map大小
        # 等价于nn.Linear(64*3*3, 128)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),  # 128x1x1
            nn.ReLU(inplace=True),
        )
        # 等价于nn.Linear(128, 10)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = torch.flatten(x, 1)
        return x


if __name__ == '__main__':
    # test net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10
    model = mnist_cnn(num_classes).to(device)
    img = torch.rand(1, 3, 28, 28).to(device)
    y = model(img)
    summary(model, input_size=(3, 28, 28))

    print(model)
    print(y)

    fcn_model = mnist_fcn(num_classes).to(device)
    img2 = torch.rand(1, 3, 28, 28).to(device)
    y2 = fcn_model(img2)
    summary(fcn_model, input_size=(3, 28, 28))
    print(fcn_model)
    print(y2)
