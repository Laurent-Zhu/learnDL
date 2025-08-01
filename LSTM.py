import torch
from torch import nn
from d2l import torch as d2l

# 长短期记忆网络 LSTM
# 引入记忆元，设置有输入门、输出门、遗忘门，同样有助于缓解梯度消失、梯度爆炸的问题，有利于获取序列中的长距离依赖关系

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
num_inputs = vocab_size
lstm_layer = nn.LSTM(num_inputs, num_hiddens)
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)

d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)