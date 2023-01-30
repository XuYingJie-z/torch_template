

class Accumulator:  # @save
    """一个累加器工具，可以同时保存和操作多个变量"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        """对保存的所有变量同时进行操作"""
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        """全部归0"""
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

