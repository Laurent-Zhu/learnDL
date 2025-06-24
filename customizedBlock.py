import torch
from torch import nn
from torch.nn import functional as F

# 自定义块
# 对于反向传播函数和参数初始化，我们不必关心，系统会自动生成
class MLP(nn.Module):
    # 构造函数中声明块中包含的层结构
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    # 定义前向传播函数
    def forward(self, X):
        # 这里使用relu的函数版本nn.functional.relu，非函数版本是nn.ReLU
        return self.out(F.relu(self.hidden(X)))

X = torch.rand(2, 20)
net = MLP()
print(net(X))

# 顺序块
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module
    
    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X
    
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print(net(X))
