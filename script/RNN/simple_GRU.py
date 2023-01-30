##############################
# GRU 简洁实现
##############################

import torch
from torch import nn
from d2l import torch as d2l
import re
import os
import MyTemplate.RNN 
from MyTemplate import RNN


## 载入数据
batch_size, num_steps = 32, 35
train_iter, vocab = MyTemplate.RNN.load_data_time_machine(batch_size, num_steps)

## 定义模型
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1

num_inputs = vocab_size
gru_layer = nn.GRU(num_inputs, num_hiddens)
model = d2l.RNNModel(gru_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)


list()