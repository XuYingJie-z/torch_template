##############################
# 处理文本数据
##############################

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB['time_machine'] = (DATA_URL + 'timemachine.txt',
                            '090b5e7e70c295757f55df93cb0a180b9691891a')

import torch
from torch import nn
from d2l import torch as d2l
import random
import re
import requests
import collections
import os
import hashlib


## 数据预处理
class DownloadTimeMachine:
    """ 下载 time machine 的文本"""

    def __init__(self) -> None:
        """不做任何事"""
        pass

    def download_time_machine(self):
        """Load the time machine dataset into a list of text lines.
        Defined in :numref:`sec_text_preprocessing`"""
        with open(self.download('time_machine'), 'r') as f:
            lines = f.readlines()
        return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

    def download(self, name, cache_dir=os.path.join('..', 'data')):
        """Download a file inserted into DATA_HUB, return the local filename.
        Defined in :numref:`sec_kaggle_house`
        cache_dir： 文件保存的 dir
        """
        assert name in DATA_HUB, f"{name} does not exist in {DATA_HUB}."
        url, sha1_hash = DATA_HUB[name]
        os.makedirs(cache_dir, exist_ok=True)
        fname = os.path.join(cache_dir, url.split('/')[-1])
        if os.path.exists(fname):
            sha1 = hashlib.sha1()
            with open(fname, 'rb') as f:
                while True:
                    data = f.read(1048576)
                    if not data:
                        break
                    sha1.update(data)
            if sha1.hexdigest() == sha1_hash:
                return fname  # Hit cache
        print(f'Downloading {fname} from {url}...')
        r = requests.get(url, stream=True, verify=True)
        with open(fname, 'wb') as f:
            f.write(r.content)
        return fname


class TextProcess:
    """ 处理文本数据"""

    def __init__(self, data_content) -> None:
        self.content = data_content

    def process(self, max_tokens):
        """文本处理函数"""
        tokens = self.tokenize(self.content, 'char')
        vocab = Vocab(tokens)
        # Since each text line in the time machine dataset is not necessarily a
        # sentence or a paragraph, flatten all the text lines into a single list
        corpus = [vocab[token] for line in tokens for token in line]
        if max_tokens > 0:
            corpus = corpus[:max_tokens]
        return corpus, vocab

    def tokenize(self, lines, token='word'):
        """Split text lines into word or character tokens.

        Defined in :numref:`sec_text_preprocessing`"""
        if token == 'word':
            return [line.split() for line in lines]
        elif token == 'char':
            return [list(line) for line in lines]
        else:
            print('ERROR: unknown token type: ' + token)




class Vocab:
    """Vocabulary for text."""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """Defined in :numref:`sec_text_preprocessing`"""
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = self.count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # The index for the unknown token is 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    def count_corpus(self, tokens):
        """Count token frequencies.

        Defined in :numref:`sec_text_preprocessing`"""
        # Here `tokens` is a 1D list or 2D list
        if len(tokens) == 0 or isinstance(tokens[0], list):
            # Flatten a list of token lists into a list of tokens
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)

    @property
    def unk(self):  # Index for the unknown token
        return 0

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs





class SeqDataLoader:
    """An iterator to load sequence data."""

    def __init__(self, corpus, vocab, batch_size, num_steps, use_random_iter):
        """Defined in :numref:`sec_language_model`"""
        if use_random_iter:
            self.data_iter_fn = self.seq_data_iter_random
        else:
            self.data_iter_fn = self.seq_data_iter_sequential

        self.corpus, self.vocab = corpus, vocab
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)

    @staticmethod
    def seq_data_iter_random(corpus, batch_size, num_steps):
        """Generate a minibatch of subsequences using random sampling.

        Defined in :numref:`sec_language_model`"""
        # Start with a random offset (inclusive of `num_steps - 1`) to partition a
        # sequence
        corpus = corpus[random.randint(0, num_steps - 1):]
        # Subtract 1 since we need to account for labels
        num_subseqs = (len(corpus) - 1) // num_steps
        # The starting indices for subsequences of length `num_steps`
        initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
        # In random sampling, the subsequences from two adjacent random
        # minibatches during iteration are not necessarily adjacent on the
        # original sequence
        random.shuffle(initial_indices)

        def data(pos):
            # Return a sequence of length `num_steps` starting from `pos`
            return corpus[pos: pos + num_steps]

        num_batches = num_subseqs // batch_size
        for i in range(0, batch_size * num_batches, batch_size):
            # Here, `initial_indices` contains randomized starting indices for
            # subsequences
            initial_indices_per_batch = initial_indices[i: i + batch_size]
            X = [data(j) for j in initial_indices_per_batch]
            Y = [data(j + 1) for j in initial_indices_per_batch]
            yield torch.tensor(X), torch.tensor(Y)

    @staticmethod
    def seq_data_iter_sequential(corpus, batch_size, num_steps):
        """Generate a minibatch of subsequences using sequential partitioning.

        Defined in :numref:`sec_language_model`"""
        # Start with a random offset to partition a sequence
        offset = random.randint(0, num_steps)
        num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
        Xs = torch.tensor(corpus[offset: offset + num_tokens])
        Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
        Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
        num_batches = Xs.shape[1] // num_steps
        for i in range(0, num_steps * num_batches, num_steps):
            X = Xs[:, i: i + num_steps]
            Y = Ys[:, i: i + num_steps]
            yield X, Y


def load_data_time_machine(batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    """
    处理文本数据的主函数，处理不同的文本数据，修改这个函数就好了
    Return the iterator and the vocabulary of the time machine dataset.
    Defined in :numref:`sec_language_model`"""
    # 下载数据
    data_downloader = DownloadTimeMachine()
    time_machine_content = data_downloader.download_time_machine()

    # 预处理
    time_machine_processor = TextProcess(time_machine_content)
    timemachine_corpus,  timemachine_vocab= time_machine_processor.process(max_tokens)
    data_iter = SeqDataLoader(
        timemachine_corpus, timemachine_vocab, batch_size, num_steps, use_random_iter
    )
    return data_iter, data_iter.vocab



def main():
    batch_size, num_steps = 32, 35
    train_iter, vocab = load_data_time_machine(batch_size, num_steps)
    print(list(train_iter)[0], vocab)

if __name__ == '__main__':
    main()