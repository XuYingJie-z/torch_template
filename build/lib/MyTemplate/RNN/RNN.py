######################################
# 从头实现 RNN 的代码
######################################



import torch
import torch.nn.functional as F
from MyTemplate import utils


class RNNModelScratch: 
    """RNN 的模型框架，重点是传入的 forward_fn，也就是RNN计算子
    """
    def __init__(self, vocab_size, num_hiddens, device,
                   forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = self.get_params(vocab_size, num_hiddens, device)
        self.forward_fn = forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32) # X.T：(num_step, batch_size)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_rnn_state(batch_size, self.num_hiddens, device)

    @staticmethod
    def get_params(vocab_size, num_hiddens, device):
        """初始化模型参数"""
        num_inputs = num_outputs = vocab_size

        def normal(shape):
            return torch.randn(size=shape, device=device) * 0.01

        # 隐藏层参数
        W_xh = normal((num_inputs, num_hiddens))
        W_hh = normal((num_hiddens, num_hiddens))
        b_h = torch.zeros(num_hiddens, device=device)
        # 输出层参数
        W_hq = normal((num_hiddens, num_outputs))
        b_q = torch.zeros(num_outputs, device=device)
        # 附加梯度
        params = [W_xh, W_hh, b_h, W_hq, b_q]
        for param in params:
            param.requires_grad_(True)
        return params

    @staticmethod
    def init_rnn_state(batch_size, num_hiddens, device):
        return (torch.zeros((batch_size, num_hiddens), device=device), )


def main():
    """ 测试 RNN，这里前向计算一下"""
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
    
    X = torch.arange(10).reshape((2, 5))
    num_hiddens = 512
    vocab_len = 28
    net = RNNModelScratch(vocab_len, num_hiddens, utils.functional.try_gpu(), rnn)
    # begin_state 传入 batch_size 即可，num_hidden上面已经传入了
    state = net.begin_state(X.shape[0], utils.functional.try_gpu()) 
    Y, new_state = net(X.to(utils.functional.try_gpu()), state)
    print(Y.shape, len(new_state), new_state[0].shape)

if __name__ == "__main__":
    main()
