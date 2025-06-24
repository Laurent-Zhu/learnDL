import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

# 本节实现权重衰减，用于解决过拟合的问题
# 权重衰减的核心是在损失函数上面加上权重的L2范数（或者其他），使L2范数尽可能小来降低模型复杂度，进而避免过拟合
# 模型复杂度较高（参数较多或者参数取值范围较大）但是数据集小，或者特征数量多而数据集小，容易导致过拟合，可以理解为背答案

# 生成训练数据
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置参数不设置权重衰减
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay':wd},
        {"params":net[0].bias}], lr=lr)
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            animator.add(epoch+1, (d2l.evaluate_loss(net, train_iter, loss), d2l.evaluate_loss(net, test_iter, loss)))
        print('w的L2范数: ', net[0].weight.norm().item())
    plt.show()

train_concise(3)
