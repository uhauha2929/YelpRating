# -*- coding: utf-8 -*-
# @Time    : 2019/12/26 9:39
# @Author  : uhauha2929


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


conf = Config(
    embedding_size=200,

    hidden_size=128,
    learning_rate=1e-3,

    batch_size=64,
    epoch=50
)

conf_elmo = Config(
    options_file='/home/yzhao/data/elmo/elmo_2x1024_128_2048cnn_1xhighway_options.json',
    weight_file='/home/yzhao/data/elmo/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5',

    hidden_size=128,
    learning_rate=1e-3,

    batch_size=64,
    epoch=50
)

conf_bert = Config(
    bert_vocab='/home/yzhao/data/bert/bert-base-uncased/vocab.txt',
    bert_dir="/home/yzhao/data/bert/bert-base-uncased/bert-base-uncased.tar.gz",

    bert_dim=768,

    hidden_size=128,
    learning_rate=1e-3,

    batch_size=10,
    epoch=50
)
