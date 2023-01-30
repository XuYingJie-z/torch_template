## 测试 functional 中的函数

import torch
from torch import nn
from MyTemplate.utils.functional import sequence_mask, masked_softmax

def test_sequence_mask():
    """ 测试 sequnce mask"""
    # 这里有两个输入，维度不一样，测试一下。
    # 结果就是 valid_len 总是作用在第二个维度上
    X = torch.tensor([[1, 2, 3], [4, 5, 6]])
    res1 = sequence_mask(X, torch.tensor([1, 2]))
    X = torch.ones(2, 3, 4)
    res2 = sequence_mask(X, torch.tensor([1, 2]), value=-1)
    return res1, res2

def test_masked_softmax():
    """测试带遮蔽的masked_softmax"""
    # 结果就是在 mask 的基础上再进行了 softmax
    return masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))


if __name__ == "__main__":
    ## 测试 sequnce mask
    test_sequence_mask()
    test_masked_softmax()






