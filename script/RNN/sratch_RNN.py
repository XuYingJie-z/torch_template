######################################
# 循环神经网络从零实现
######################################


import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from MyTemplate import utils, RNN
from MyTemplate.RNN import RNNscratch



def rnn(inputs, state, params):
    """RNN 操作子"""
    # inputs的形状：(时间步数量，批量大小，词表大小)
    # state.shape=(batch, num_hidden)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形状：(批量大小，词表大小)
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h) # 这里都是 batch_size x num_hidden
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)




def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    # 预热期，就是把 prefix 里的句子过一遍。循环之后 outpus = prefix
    for y in prefix[1:]:  
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    # 正式预测
    for _ in range(num_preds):  # 预测num_preds步
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])



class TrainRNN:
    def __init__(self, net, vocab, train_iter, loss, updater, device, use_random_iter) -> None:
        self.net = net
        self.train_iter = train_iter
        self.vocab = vocab
        self.loss = loss
        self.updater = updater
        self.device = device
        self.use_random_iter = use_random_iter
                

    def train_one_epoch(self):
        """训练网络一个迭代周期（定义见第8章）"""
        state, timer = None, d2l.Timer()
        metric = utils.Accumulator(2)  # 训练损失之和,词元数量
        for X, Y in self.train_iter:
            if state is None or self.use_random_iter:
                # 在第一次迭代或使用随机抽样时初始化state
                state = self.net.begin_state(batch_size=X.shape[0], device=self.device)
            else:
                if isinstance(self.net, nn.Module) and not isinstance(state, tuple):
                    # state对于nn.GRU是个张量
                    state.detach_()
                else:
                    # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                    for s in state:
                        s.detach_()
            y = Y.T.reshape(-1)
            X, y = X.to(self.device), y.to(self.device)
            y_hat, state = self.net(X, state)
            l = self.loss(y_hat, y.long()).mean()
            if isinstance(self.updater, torch.optim.Optimizer):
                # pytorch 框架的更新方法
                self.updater.zero_grad()
                l.backward()
                self.grad_clipping(self.net, 1)
                self.updater.step()
            else:
                # 自己动手实现的更新方法
                l.backward()
                self.grad_clipping(self.net, 1)
                # 因为已经调用了mean函数
                self.updater(batch_size=1)
            metric.add(l * y.numel(), y.numel())
        return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

    def train_multi_epoch(self, num_epochs):
        """训练多轮"""
        animator = utils.Animator(xlabel='epoch', ylabel='perplexity',
                                legend=['train'], xlim=[10, num_epochs])
        predict = lambda prefix: predict_ch8(prefix, 50, self.net, self.vocab, self.device)
        # 训练和预测
        for epoch in range(num_epochs):
            ppl, speed = self.train_one_epoch()  # 训练一个 epoch
            if (epoch + 1) % 10 == 0:
                print(predict('time traveller'))
                animator.add(epoch + 1, [ppl])
        print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(self.device)}')
        print(predict('time traveller'))
        print(predict('traveller'))
    
    
    @staticmethod
    def grad_clipping(net, theta):  #@save
        """裁剪梯度"""
        if isinstance(net, nn.Module):
            params = [p for p in net.parameters() if p.requires_grad]
        else:
            params = net.params # TODO 可以改一下，
        # 这里就是算 L2 范数
        norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
        if norm > theta:
            for param in params:
                param.grad[:] *= theta / norm


def main():
    ## 读取数据
    batch_size, num_steps = 32, 35
    train_iter, vocab = RNN.TextDataPrepare.load_data_time_machine(batch_size, num_steps)

    ## 定义损失函数
    loss = nn.CrossEntropyLoss()
    updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size) # 自己动手实现的优化器

    # 定义学习率
    num_epochs, lr, use_random_iter = 500, 1, False
    num_hiddens = 512
    net = RNN.RNNModelScratch(len(vocab), num_hiddens, utils.functional.try_gpu(), rnn)

    # 开始训练
    RNN_trainer = TrainRNN(net, vocab, train_iter, loss, updater, utils.functional.try_gpu(), use_random_iter)
    RNN_trainer.train_multi_epoch(10) # 训练多少epoach

if __name__ == "__main__":
    main()
