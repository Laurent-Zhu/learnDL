import torch
from torch import nn

# 参数管理：参数访问、参数初始化、参数绑定

# 1.参数访问
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
print(net(X))
# 可以通过索引访问任意层并查看该层的参数
print(net[2].state_dict())
# 输出: OrderedDict([('weight', tensor([[-0.2486,  0.2330, -0.2726, -0.0584,  0.2432, -0.2458, -0.2898, -0.0796]])), ('bias', tensor([-0.3133]))])
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.data)
# 一次性访问所有参数
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])

print(net.state_dict()['2.bias'].data)

# 从嵌套块中收集参数
def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}', block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)
print(rgnet)
print(rgnet[0][1][0].bias.data)     # 访问第一个主要的块中的第二个子块的第一层偏置

# 2.参数初始化
# 可以通过pytorch提供的默认初始化，也可以自定义
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
print(f'初始化init_normal: net[0].weight.data[0] = {net[0].weight.data[0]}, net[0].bias.data[0] = {net[0].bias.data[0]}')

def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
print(f'初始化init_constant: net[0].weight.data[0] = {net[0].weight.data[0]}, net[0].bias.data[0] = {net[0].bias.data[0]}')
# 可以对不同块应用不同的初始化方法
def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data[0])

# 自定义初始化
def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
print(f'使用my_init: net[0].weight[:2] = {net[0].weight[:2]}')

# 3.参数绑定
# 用于多个层之间共享参数
shared = nn.Linear(8, 8)    # 通过定义一个稠密层来实现
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
net(X)
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0] = 100
print(net[2].weight.data[0] == net[4].weight.data[0])
