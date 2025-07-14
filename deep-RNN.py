import torch
from torch import nn
from d2l import torch as d2l

# 深度循环神经网络
# 与单层的循环神经网络相比，deepRNN具有多个隐藏层，隐状态的信息被传递到当前层的下一时间步和下一层的当前时间步

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

vocab_size, num_hiddens, num_layers = len(vocab), 256, 2
num_inputs = vocab_size
device = d2l.try_gpu()
lstm_layer = nn.LSTM(num_inputs, num_hiddens, num_layers)

model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)

num_epochs, lr = 500, 2
d2l.train_ch8(model, train_iter, vocab, lr*1.0, num_epochs, device)