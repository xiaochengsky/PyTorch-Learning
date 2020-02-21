import torch
import torchvision
from torchvision import transforms
import time
from torch import nn, optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 下载数据集 FashionMNIST 数据集
# mnist_train = torchvision.datasets.FashionMNIST(root='/mnt_datas/ycc/dataset/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
# mnist_test = torchvision.datasets.FashionMNIST(root='/mnt_datas/ycc/dataset/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())
# print(type(mnist_train))
# print(len(mnist_train), len(mnist_test))

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


net = LeNet()
# 查看网络各层大小
print(net)

