import torch
import torchvision
from torchvision import transforms
import time
from torch import nn, optim
import sys


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


# 加载 FashionMNIST 数据集
def load_data_fashion_mnist(batch_size, resize=None, root='/mnt_datas/ycc/dataset/FashionMNIST'):
    """Download the fashion mnist dataset and then load into memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_iter, test_iter


# 验证当前网络在测试集的精确度
def evaluate_accuracy(data_iter, net, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        # 验证模式, 放弃 dropOut
        net.eval()
        acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
        # 改为训练模式
        net.train()
        n += y.shape[0]
    return acc_sum / n


# LeNet网络
def train_leNet(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            i = 1
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec' %
              (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


# LeNet网络定义
net = nn.Sequential(
    nn.Conv2d(1, 6, 5),
    nn.BatchNorm2d(6),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(6, 16, 5),
    nn.BatchNorm2d(16),
    nn.Sigmoid(),
    nn.MaxPool2d(2, 2),
    FlattenLayer(),
    nn.Linear(16*4*4, 120),
    nn.BatchNorm1d(120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.BatchNorm1d(84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)
# 查看网络各层大小
print(net)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置批量大小
batch_size = 256

# 获取数据和训练数据
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)
lr, num_epochs = 0.001, 5

# 定义优化器
optimizer = torch.optim.Adam(net.parameters(), lr=lr)
train_leNet(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)

# 查看第一层的BN的 gamma 和 beta 参数
print(net[1].gamma.view((-1, )), net[1].beta.view((-1, )))
