import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import utils
import d2l.torch as d2l




## 数据封装

class TensorDataset(Dataset):
    """
    TensorDataset继承Dataset, 核心是重载 __init__(), __getitem__(), __len__()
    实现将一组Tensor数据对封装成Tensor数据集
    能够通过index得到数据集的数据。通过len，得到数据集大小
    """

    def __init__(self, data_tensor, target_tensor):
        """
        重写 init
        :param data_tensor: x 数据, 先行神经网络: data_tensor.shape=num_sample x num_feature
        :param target_tensor: y 数据，分类问题就是标号，不需要独热编码, target_tensor.shape=num_sample
        """
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        """
        获取数据
        :param index: 数据的编号
        :return: (数据, 数据对应的分类)
        """
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self) -> int:
        """
        :return: 返回数据大小
        """
        return self.data_tensor.size(0)


## 定义模型
class Line_Module(nn.Module):
    """线性神经网络模型"""
    def __init__(self) -> None:
        """
        初始化模型，后续需要的操作子都要在这里实例化
        注意： loss_func 使用 crossetropy 的话，这里就不需要用 softmax，crossetropy 自带 softmax
        测试：可以用后面的 test 进行测试，无需实例化，直接 Line_Module.test() 即可
        """
        super().__init__()
        self.line_layer_1 = nn.Linear(10, 20)
        self.line_layer_2 = nn.Linear(20, 60)
        self.line_layer_3 = nn.Linear(60, 20)
        self.line_layer_4 = nn.Linear(20, 4)
        self.dropout_layer = nn.Dropout(0.3)
        self.relu_layer = nn.ReLU()
        # self.softmax_layer = nn.Softmax(dim=1)

    def forward(self, x):
        """
        由于继承了 nn.Module，这里的代码用于模型前向运算
        model = Line_Module()
        model(x) # 这里调用的就是下面的代码
        :param x:
        :return:
        """
        res = self.relu_layer(self.dropout_layer(self.line_layer_1(x)))
        res = self.relu_layer(self.dropout_layer(self.line_layer_2(res)))
        res = self.relu_layer(self.dropout_layer(self.line_layer_3(res)))
        res = self.relu_layer(self.dropout_layer(self.line_layer_4(res)))
        # res = self.softmax_layer(res)
        return res

    @staticmethod
    def init_weight(m):
        """
        权重初始化函数，放这里是因为易于归类
        :param m:
        :return:
        """
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    @staticmethod
    def test_model():
        """
        测试代码，无需实例化，直接 Line_Module.test() 即可
        """
        model_x = Line_Module()
        Line_Module.init_weight(model_x)
        return model_x(torch.randn((20, 10)))


class TrainNet:
    """训练模型"""
    def __init__(self, model, train_dataset, test_dataset, loss, updater) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.loss = loss
        self.updater = updater

    def train_one_epoch(self):
        # 训练单个循环的代码
        # Set the model to training mode
        if isinstance(self.model, torch.nn.Module):
            self.model.train()
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = utils.Accumulator(3)
        for X, y in self.train_dataset:
            # Compute gradients and update parameters
            y_hat = self.model(X.data)  # torch 更新版本之后需要加上 x.data
            l = self.loss(y_hat, y)
            if isinstance(self.updater, torch.optim.Optimizer):
                # Using PyTorch in-built optimizer & loss criterion
                self.updater.zero_grad()
                l.mean().backward()
                self.updater.step()
            else:
                # Using !!custom built optimizer & loss criterion
                l.sum().backward()
                self.updater(X.shape[0])
            metric.add(float(l.sum()), self.accuracy(y_hat, y), y.numel())
        # Return training loss and training accuracy
        return metric[0] / metric[2], metric[1] / metric[2]

    @staticmethod
    def accuracy(y_hat, y):
        """Compute the number of correct predictions.
        Defined in :numref:`sec_softmax_scratch`"""
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(1)
        cmp = y_hat.type(y.dtype) == y
        return float(cmp.type(y.dtype).sum())

    def train_epoachs(self, num_epochs):
        """Train a model (defined in Chapter 3).

        Defined in :numref:`sec_softmax_scratch`"""
        animator = utils.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])
        for epoch in range(num_epochs):
            train_metrics = self.train_one_epoch()
            test_acc = self.evaluate_accuracy()
            if epoch == num_epochs-1:
                animator.add(epoch + 1, train_metrics + (test_acc,))

            animator.add(epoch + 1, train_metrics + (test_acc,))
        train_loss, train_acc = train_metrics
        assert train_loss < 0.5, train_loss
        assert train_acc <= 1 and train_acc > 0.7, train_acc
        assert test_acc <= 1 and test_acc > 0.7, test_acc

    def evaluate_accuracy(self):
        """Compute the accuracy for a model on a dataset.

        Defined in :numref:`sec_softmax_scratch`"""
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()  # Set the model to evaluation mode
        metric = utils.Accumulator(2)  # No. of correct predictions, no. of predictions

        with torch.no_grad():
            for X, y in self.test_dataset:
                metric.add(self.accuracy(self.model(X), y), y.numel())
        return metric[0] / metric[1]


def main() -> None:
    ## 构造数据
    x = torch.arange(4).repeat(100)
    # embedd 一下，然后加上噪音
    embedd_layer = nn.Embedding(400, 10)
    x_train = embedd_layer(x) + torch.randn((400, 10))

    y_train = torch.arange(4).repeat(100)

    # 将数据封装成Dataset 和 dataloder
    tensor_dataset = TensorDataset(x_train, y_train)

    tensor_dataloader = DataLoader(tensor_dataset,  # 封装的对象
                                   batch_size=40,  # 输出的batch size
                                   shuffle=True,  # 随机输出
                                   num_workers=0)  # 只有1个进程

    model_x = Line_Module()
    Line_Module.init_weight(model_x) # 初始化权重
    loss = nn.CrossEntropyLoss(reduction='none')
    updater = torch.optim.SGD(model_x.parameters(), lr=0.005)

    train_obj = TrainNet(model_x, tensor_dataloader, tensor_dataloader, loss, updater)
    res = train_obj.train_epoachs(500)

if __name__ == '__main__':
    main()