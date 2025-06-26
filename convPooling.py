import torch
from torch import nn
from d2l import torch as d2l

# 池化层/汇聚层 Pooling
# 起到汇聚、聚拢信息的作用：降低卷积层对位置的敏感性；降低对空间降采样表示的敏感性
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))

pool2d = nn.MaxPool2d(3)
print(pool2d(X))

pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))

pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
print(pool2d(X))
# 多通道
X = torch.cat((X, X+1), 1)
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
print(pool2d(X))