import torch
from d2l import torch as d2l
import torch.nn as nn
import matplotlib.pyplot as plt

# 本节实现暂退法
# 暂退法是以p的概率丢弃隐藏层单元，只需要在每个全连接层之后添加一个暂退层

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
dropout1, dropout2 = 0.2, 0.5   # 两个暂退层的概率
num_epochs, lr, batch_size = 10, 0.5, 256

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    # 在第一个全连接层之后添加一个暂退层
    nn.Dropout(dropout1),
    nn.Linear(256, 256),
    nn.ReLU(),
    # 在第二个全连接层之后添加一个暂退层
    nn.Dropout(dropout2),
    nn.Linear(256, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

trainer = torch.optim.SGD(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
plt.show()