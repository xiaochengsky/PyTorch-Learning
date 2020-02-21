import torch
import torchvision
from torchvision import transforms

mnist_train = torchvision.datasets.FashionMNIST(root='/mnt_datas/ycc/dataset/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())

mnist_test = torchvision.datasets.FashionMNIST(root='/mnt_datas/ycc/dataset/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())

print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

