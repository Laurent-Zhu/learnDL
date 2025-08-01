import torch
from torch import nn
from d2l import torch as d2l

# 本节实现卷积运算的基本操作
def corr2d(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

# 卷积层被训练的两个参数就是卷积核和标量偏置
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

X = torch.ones((6, 8))
X[:, 2:6] = 0
print(X)
K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)
print(Y)
Z = corr2d(X.t(), K.t())
print(Z)

# 学习卷积核：通过输入和输出来迭代更新卷积核
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False) # 批量大小，通道数，卷积核大小
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2

for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y) ** 2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    print(f'epoch {i+1}, loss {l.sum(): .3f}')

print(conv2d.weight.data.reshape((1, 2)))