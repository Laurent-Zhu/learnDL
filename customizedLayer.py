import torch
from torch import nn
import torch.nn.functional as F

# 自定义层
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units, ))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias. data
        return F.relu(linear)

linear = MyLinear(5, 3)
print(linear(torch.rand(2, 5)))
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
print(net(torch.rand(2, 64)))